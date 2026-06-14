import os
import io
import re
import json
import uuid
import codecs
import httpx
import string
import secrets
import tomllib
import asyncio
import random
from asyncio import Semaphore
import contextvars
from time import time
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from collections import defaultdict
from contextlib import aclosing, asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Dict, Union, Optional, List, Any, Awaitable, Callable
from pydantic import ValidationError, BaseModel, field_serializer

from starlette.responses import Response
from starlette.types import Scope, Receive, Send
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse as StarletteStreamingResponse

from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from fastapi import FastAPI, HTTPException, Depends, Request, Body, BackgroundTasks, UploadFile, File, Form, Query

from core.log_config import logger, trace_logger
from core.request import (
    CODEX_CLI_VERSION,
    CODEX_USER_AGENT,
    apply_post_body_parameter_overrides,
    force_codex_client_headers,
    get_payload,
    strip_unsupported_codex_payload_fields,
)
from core.response import fetch_response, fetch_response_stream
from core.models import RequestModel, ResponsesRequest, ImageGenerationRequest, ImageEditRequest, AudioTranscriptionRequest, ModerationRequest, TextToSpeechRequest, UnifiedRequest, EmbeddingRequest
from core.utils import (
    get_proxy,
    get_engine,
    parse_rate_limit,
    IncrementalSSEParser,
    parse_sse_event,
    collect_openai_chat_completion_from_streaming_sse,
    ThreadSafeCircularList,
    provider_api_circular_list,
)
from routing import (
    RoutingPlan,
    build_api_key_model_response_cache,
    build_api_key_models_map,
    build_routing_index,
    _call_provider_resolver,
    estimate_request_total_tokens,
    get_right_order_providers,
    select_provider_api_key_raw,
)
from upstream import (
    UPSTREAM_NETWORK_ERRORS,
    UpstreamRunner,
    build_upstream_error_response,
)
from fugue_observability import (
    emit_uni_api_ember_request_observability,
    start_fugue_observability_from_env,
    stop_fugue_observability,
)
from video import VideoAdapterError, get_video_adapter

from utils import (
    safe_get,
    load_config,
    update_config,
    post_all_models,
    InMemoryRateLimiter,
    error_handling_wrapper,
    query_channel_key_stats,
)

from sqlalchemy import inspect, text
from sqlalchemy.sql import sqltypes
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, case, func, desc

from db import Base, RequestStat, ChannelStat, db_engine, async_session, DISABLE_DATABASE

def _env_flag(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


def _should_log_stdout_request_summary() -> bool:
    if not _env_bool("STDOUT_REQUEST_SUMMARY_LOG_ENABLED", True):
        return False
    sample_rate = max(0.0, min(1.0, _env_float("STDOUT_REQUEST_SUMMARY_LOG_SAMPLE_RATE", 1.0)))
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    return random.random() <= sample_rate


def _log_stdout_request_summary(provider: str, model: str, engine: str, role: str) -> None:
    if not _should_log_stdout_request_summary():
        return
    logger.info(
        "provider: %-11s model: %-22s engine: %-13s role: %s",
        str(provider or "")[:11],
        str(model or ""),
        str(engine or "")[:13],
        role,
    )


DEFAULT_TIMEOUT = int(os.getenv("TIMEOUT", 100))
is_debug = _env_flag(os.getenv("DEBUG"))
logger.info("DISABLE_DATABASE: %s", DISABLE_DATABASE)

_REQUEST_ID_RE = re.compile(r"[^A-Za-z0-9_.:-]")
_W3C_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_W3C_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-(?P<span_id>[0-9a-f]{16})-(?P<trace_flags>[0-9a-f]{2})(?:-.*)?$",
    re.IGNORECASE,
)


def _normalize_request_id(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return str(uuid.uuid4())
    normalized = _REQUEST_ID_RE.sub("-", raw)[:96].strip("-")
    return normalized or str(uuid.uuid4())


def _is_valid_w3c_trace_id(value: Optional[str]) -> bool:
    trace_id = str(value or "").strip().lower()
    return bool(_W3C_TRACE_ID_RE.match(trace_id)) and trace_id != "0" * 32


def _is_valid_w3c_span_id(value: Optional[str]) -> bool:
    span_id = str(value or "").strip().lower()
    return bool(_W3C_SPAN_ID_RE.match(span_id)) and span_id != "0" * 16


def _parse_traceparent(value: Optional[str]) -> dict[str, str]:
    raw = str(value or "").strip()
    match = _TRACEPARENT_RE.match(raw)
    if not match:
        return {}
    version = match.group("version").lower()
    trace_id = match.group("trace_id").lower()
    span_id = match.group("span_id").lower()
    trace_flags = match.group("trace_flags").lower()
    if version == "ff" or not _is_valid_w3c_trace_id(trace_id) or not _is_valid_w3c_span_id(span_id):
        return {}
    return {
        "trace_id": trace_id,
        "parent_span_id": span_id,
        "trace_flags": trace_flags,
    }


def _incoming_trace_context(headers: Any) -> dict[str, str]:
    parsed = _parse_traceparent(headers.get("traceparent") if headers else None)
    raw_legacy_request_id = str(headers.get("x-request-id") or "").strip() if headers else ""
    legacy_request_id = _normalize_request_id(raw_legacy_request_id) if raw_legacy_request_id else ""
    if parsed:
        result = dict(parsed)
        if legacy_request_id and legacy_request_id != result["trace_id"]:
            result["x_request_id"] = legacy_request_id
        tracestate = str(headers.get("tracestate") or "").strip() if headers else ""
        if tracestate:
            result["tracestate"] = tracestate[:512]
        return result
    if legacy_request_id:
        return {"trace_id": legacy_request_id}
    return {"trace_id": uuid.uuid4().hex}


def _format_traceparent(trace_id: Optional[str], span_id: Optional[str], trace_flags: Optional[str] = None) -> Optional[str]:
    safe_trace_id = str(trace_id or "").strip().lower()
    safe_span_id = str(span_id or "").strip().lower()
    if not _is_valid_w3c_trace_id(safe_trace_id) or not _is_valid_w3c_span_id(safe_span_id):
        return None
    safe_flags = str(trace_flags or "01").strip().lower()
    if not re.match(r"^[0-9a-f]{2}$", safe_flags):
        safe_flags = "01"
    return f"00-{safe_trace_id}-{safe_span_id}-{safe_flags}"


class RequestTrace:
    def __init__(
        self,
        *,
        trace_id: str,
        parent_span_id: Optional[str] = None,
        trace_flags: Optional[str] = None,
        tracestate: Optional[str] = None,
    ) -> None:
        self.trace_id = _normalize_request_id(trace_id)
        self.span_id = secrets.token_hex(8)
        self.parent_span_id = str(parent_span_id or "").strip().lower()
        self.trace_flags = str(trace_flags or "01").strip().lower()
        self.tracestate = str(tracestate or "").strip()
        self.started_at = time()
        self.spans: dict[str, int | str] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }
        if self.parent_span_id:
            self.spans["parent_span_id"] = self.parent_span_id

    def mark(self, stage: str) -> None:
        name = str(stage or "").strip()
        if name:
            self.spans[name] = int((time() - self.started_at) * 1000)

    def add_ms(self, name: str, value_ms: float) -> None:
        key = str(name or "").strip()
        if not key:
            return
        try:
            self.spans[key] = max(0, int(round(float(value_ms))))
        except (TypeError, ValueError):
            return

    def set_tag(self, name: str, value: Optional[str]) -> None:
        key = str(name or "").strip()
        text = str(value or "").strip()
        if key and text:
            self.spans[key] = text[:128]

    def snapshot(self) -> dict[str, int | str]:
        return dict(self.spans)


class RuntimeGauges:
    def __init__(self) -> None:
        self.inflight_requests = 0
        self.waiting_first_byte_requests: set[str] = set()
        self.waiting_first_byte_untracked = 0
        self.event_loop_lag_ms = 0
        self.open_sockets: Optional[int] = None
        self.upstream_pool_in_use = 0
        self.upstream_pool_wait_ms = 0

    def begin_inflight(self) -> None:
        self.inflight_requests += 1

    def end_inflight(self) -> None:
        self.inflight_requests = max(0, self.inflight_requests - 1)

    def _request_info(self, current_info: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        if isinstance(current_info, dict):
            return current_info
        try:
            info = request_info.get()
        except LookupError:
            return None
        return info if isinstance(info, dict) else None

    def _request_key(self, current_info: Optional[dict[str, Any]] = None) -> Optional[str]:
        info = self._request_info(current_info)
        if not info:
            return None
        key = str(info.get("request_id") or info.get("trace_id") or "").strip()
        return key or None

    def begin_waiting_first_byte(self, current_info: Optional[dict[str, Any]] = None) -> None:
        info = self._request_info(current_info)
        key = self._request_key(info)
        if key:
            self.waiting_first_byte_requests.add(key)
            if info is not None:
                info["_waiting_first_byte_active"] = True
            return
        self.waiting_first_byte_untracked += 1

    def end_waiting_first_byte(self, current_info: Optional[dict[str, Any]] = None) -> None:
        info = self._request_info(current_info)
        key = self._request_key(info)
        if key:
            self.waiting_first_byte_requests.discard(key)
            if info is not None:
                info["_waiting_first_byte_active"] = False
            return
        self.waiting_first_byte_untracked = max(0, self.waiting_first_byte_untracked - 1)

    def begin_upstream_pool(self, trace: Optional[RequestTrace] = None) -> float:
        started_at = time()
        self.upstream_pool_in_use += 1
        wait_ms = int((time() - started_at) * 1000)
        self.upstream_pool_wait_ms = wait_ms
        if trace is not None:
            trace.add_ms("upstream_pool_wait_ms", wait_ms)
        return started_at

    def end_upstream_pool(self) -> None:
        self.upstream_pool_in_use = max(0, self.upstream_pool_in_use - 1)

    async def record_event_loop_lag(self) -> None:
        started_at = time()
        await asyncio.sleep(0)
        self.event_loop_lag_ms = int((time() - started_at) * 1000)

    def snapshot(self) -> dict[str, Any]:
        self.open_sockets = _open_socket_count()
        tcp_states = _tcp_state_counts()
        return {
            "service": "uni-api-ember",
            "inflight_requests": self.inflight_requests,
            "waiting_first_byte": len(self.waiting_first_byte_requests) + self.waiting_first_byte_untracked,
            "event_loop_lag_ms": self.event_loop_lag_ms,
            "open_sockets": self.open_sockets,
            "tcp_states": tcp_states,
            "tcp_close_wait": tcp_states.get("CLOSE_WAIT", 0),
            "upstream_pool_in_use": self.upstream_pool_in_use,
            "upstream_pool_wait_ms": self.upstream_pool_wait_ms,
        }


runtime_gauges = RuntimeGauges()


def _open_socket_count() -> Optional[int]:
    fd_dir = "/proc/self/fd"
    if not os.path.isdir(fd_dir):
        return None
    count = 0
    try:
        for name in os.listdir(fd_dir):
            try:
                if os.readlink(os.path.join(fd_dir, name)).startswith("socket:"):
                    count += 1
            except OSError:
                continue
    except OSError:
        return None
    return count


_TCP_STATES = {
    "01": "ESTABLISHED",
    "02": "SYN_SENT",
    "03": "SYN_RECV",
    "04": "FIN_WAIT1",
    "05": "FIN_WAIT2",
    "06": "TIME_WAIT",
    "07": "CLOSE",
    "08": "CLOSE_WAIT",
    "09": "LAST_ACK",
    "0A": "LISTEN",
    "0B": "CLOSING",
}


_BACKGROUND_CLEANUP_TASKS: set[asyncio.Task[Any]] = set()


def _drain_current_task_cancellation() -> None:
    current_task = asyncio.current_task()
    uncancel = getattr(current_task, "uncancel", None)
    if callable(uncancel):
        while current_task is not None and current_task.cancelling():
            uncancel()


def _track_background_cleanup_task(task: asyncio.Task[Any], *, label: str) -> None:
    _BACKGROUND_CLEANUP_TASKS.add(task)

    def _cleanup_done(done: asyncio.Task[Any]) -> None:
        _BACKGROUND_CLEANUP_TASKS.discard(done)
        if done.cancelled():
            logger.warning("%s cleanup task was cancelled after detach", label)
            return
        try:
            done.result()
        except BaseException as exc:
            logger.warning(
                "%s cleanup failed after detach",
                label,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    task.add_done_callback(_cleanup_done)


def _tcp_state_counts() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for path in ("/proc/self/net/tcp", "/proc/self/net/tcp6"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                rows = handle.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            parts = row.split()
            if len(parts) < 4:
                continue
            state = _TCP_STATES.get(parts[3].upper(), parts[3].upper())
            counts[state] += 1
    return dict(counts)


def _tcp_close_wait_socket_inodes() -> set[str]:
    inodes: set[str] = set()
    for path in ("/proc/self/net/tcp", "/proc/self/net/tcp6"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                rows = handle.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            parts = row.split()
            if len(parts) <= 9:
                continue
            if parts[3].upper() == "08":
                inode = parts[9].strip()
                if inode and inode != "0":
                    inodes.add(inode)
    return inodes


def _socket_inode_for_fd(fd: int) -> Optional[str]:
    try:
        target = os.readlink(f"/proc/self/fd/{int(fd)}")
    except (OSError, TypeError, ValueError):
        return None
    match = re.match(r"^socket:\[(\d+)\]$", target)
    return match.group(1) if match else None


def _httpcore_connection_socket_inode(connection: Any) -> Optional[str]:
    inner_connection = getattr(connection, "_connection", None) or connection
    network_stream = getattr(inner_connection, "_network_stream", None)
    get_extra_info = getattr(network_stream, "get_extra_info", None)
    if not callable(get_extra_info):
        return None
    try:
        sock = get_extra_info("socket")
    except BaseException:
        return None
    fileno = getattr(sock, "fileno", None)
    if not callable(fileno):
        return None
    try:
        fd = fileno()
    except BaseException:
        return None
    return _socket_inode_for_fd(fd)


async def _await_cleanup_safely(awaitable: Any, *, label: str) -> bool:
    if awaitable is None or not hasattr(awaitable, "__await__"):
        return True

    _drain_current_task_cancellation()

    cleanup_task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.shield(cleanup_task)
        return True
    except asyncio.CancelledError as exc:
        logger.warning(
            "%s cleanup was cancelled; waiting for cleanup to finish",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        _drain_current_task_cancellation()
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as final_exc:
            _drain_current_task_cancellation()
            logger.warning(
                "%s cleanup was cancelled again; detached cleanup will continue",
                label,
                exc_info=(type(final_exc), final_exc, final_exc.__traceback__),
            )
            _track_background_cleanup_task(cleanup_task, label=label)
            return True
        except GeneratorExit as final_exc:
            logger.warning(
                "%s cleanup was interrupted by generator close; detached cleanup will continue",
                label,
                exc_info=(type(final_exc), final_exc, final_exc.__traceback__),
            )
            _track_background_cleanup_task(cleanup_task, label=label)
            return True
        except BaseException as final_exc:
            logger.warning(
                "%s cleanup failed after cancellation",
                label,
                exc_info=(type(final_exc), final_exc, final_exc.__traceback__),
            )
            return False
        return True
    except GeneratorExit as exc:
        logger.warning(
            "%s cleanup was interrupted by generator close; detached cleanup will continue",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        _track_background_cleanup_task(cleanup_task, label=label)
        return True
    except BaseException as exc:
        logger.warning(
            "%s cleanup failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False


async def _call_cleanup_safely(cleanup: Callable[[], Any], *, label: str) -> bool:
    try:
        result = cleanup()
    except BaseException as exc:
        logger.warning(
            "%s cleanup failed",
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return await _await_cleanup_safely(result, label=label)


async def _force_release_httpcore_pool_request_safely(stream: Any) -> bool:
    pool = getattr(stream, "_pool", None)
    pool_request = getattr(stream, "_pool_request", None)
    if pool is None or pool_request is None:
        return True

    requests = getattr(pool, "_requests", None)
    pool_connections = getattr(pool, "_connections", None)
    connection = getattr(pool_request, "connection", None)
    if not isinstance(requests, list):
        requests = []
    if pool_request not in requests and connection is None:
        return True

    try:
        closing: list[Any] = []
        lock = getattr(pool, "_optional_thread_lock", None)
        if lock is not None:
            with lock:
                if pool_request in requests:
                    requests.remove(pool_request)
                if isinstance(pool_connections, list) and connection in pool_connections:
                    pool_connections.remove(connection)
                assign_requests = getattr(pool, "_assign_requests_to_connections", None)
                closing = list(assign_requests()) if callable(assign_requests) else closing
        else:
            if pool_request in requests:
                requests.remove(pool_request)
            if isinstance(pool_connections, list) and connection in pool_connections:
                pool_connections.remove(connection)
            assign_requests = getattr(pool, "_assign_requests_to_connections", None)
            closing = list(assign_requests()) if callable(assign_requests) else closing

        if connection is not None and all(candidate is not connection for candidate in closing):
            closing.append(connection)

        close_connections = getattr(pool, "_close_connections", None)
        if callable(close_connections):
            return await _await_cleanup_safely(
                close_connections(closing),
                label="Upstream HTTP pool request",
            )
        cleanup_ok = True
        for connection_to_close in closing:
            aclose = getattr(connection_to_close, "aclose", None)
            if callable(aclose):
                cleanup_ok = await _call_cleanup_safely(
                    aclose,
                    label="Upstream HTTP pool request connection",
                ) and cleanup_ok
        return cleanup_ok
    except BaseException as exc:
        logger.warning(
            "Upstream HTTP pool request cleanup failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False
    return True


async def _force_close_httpcore_stream_chain_safely(upstream_response: Any) -> bool:
    stream = getattr(upstream_response, "stream", None)
    candidates: list[Any] = []
    current = stream
    seen: set[int] = set()
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        candidates.append(current)
        current = getattr(current, "_stream", None)

    cleanup_ok = True
    for candidate in candidates:
        aclose = getattr(candidate, "aclose", None)
        if callable(aclose):
            cleanup_ok = await _call_cleanup_safely(
                aclose,
                label="Upstream HTTP response stream",
            ) and cleanup_ok
        cleanup_ok = await _force_release_httpcore_pool_request_safely(candidate) and cleanup_ok
    return cleanup_ok


async def _close_upstream_response_safely(upstream_response: Any | None) -> bool:
    if upstream_response is None:
        return True

    cleanup_ok = True
    aclose = getattr(upstream_response, "aclose", None)
    if callable(aclose):
        cleanup_ok = await _call_cleanup_safely(
            aclose,
            label="Upstream HTTP response",
        ) and cleanup_ok
    cleanup_ok = await _force_close_httpcore_stream_chain_safely(upstream_response) and cleanup_ok
    return cleanup_ok


async def _close_stream_cm_safely(stream_cm: Any | None) -> bool:
    if stream_cm is None:
        return True
    close = getattr(stream_cm, "__aexit__", None)
    if not callable(close):
        return True
    return await _await_cleanup_safely(
        close(None, None, None),
        label="Upstream stream context manager",
    )


async def _close_upstream_response_stream_safely(
    stream_cm: Any | None,
    upstream_response: Any | None,
) -> bool:
    cleanup_ok = await _close_upstream_response_safely(upstream_response)
    cleanup_ok = await _close_stream_cm_safely(stream_cm) and cleanup_ok
    return cleanup_ok


async def _sweep_httpx_client_idle_connections(client: httpx.AsyncClient) -> int:
    transport = getattr(client, "_transport", None)
    pool = getattr(transport, "_pool", None)
    assign_requests = getattr(pool, "_assign_requests_to_connections", None)
    close_connections = getattr(pool, "_close_connections", None)
    pool_connections = getattr(pool, "_connections", None)
    if not callable(assign_requests) or not callable(close_connections):
        return 0
    if not isinstance(pool_connections, list):
        return 0

    lock = getattr(pool, "_optional_thread_lock", None)
    closing: list[Any] = []
    close_wait_inodes = _tcp_close_wait_socket_inodes()

    def collect_connection(connection: Any) -> None:
        if all(candidate is not connection for candidate in closing):
            closing.append(connection)

    def should_close_connection(connection: Any) -> bool:
        inode = _httpcore_connection_socket_inode(connection)
        if inode is not None and inode in close_wait_inodes:
            return True
        is_closed = getattr(connection, "is_closed", None)
        has_expired = getattr(connection, "has_expired", None)
        return (callable(is_closed) and is_closed()) or (callable(has_expired) and has_expired())

    if lock is not None:
        with lock:
            for connection in list(pool_connections):
                if should_close_connection(connection):
                    if connection in pool_connections:
                        pool_connections.remove(connection)
                    collect_connection(connection)
            for connection in assign_requests():
                collect_connection(connection)
    else:
        for connection in list(pool_connections):
            if should_close_connection(connection):
                if connection in pool_connections:
                    pool_connections.remove(connection)
                collect_connection(connection)
        for connection in assign_requests():
            collect_connection(connection)
    if not closing:
        return 0
    await close_connections(closing)
    return len(closing)


def _current_trace() -> Optional[RequestTrace]:
    try:
        info = request_info.get()
    except LookupError:
        return None
    trace = info.get("trace") if isinstance(info, dict) else None
    return trace if isinstance(trace, RequestTrace) else None


def _mark_stage(stage: str) -> None:
    trace = _current_trace()
    if trace is not None:
        trace.mark(stage)


def _trace_headers_for_upstream(current_info: dict[str, Any]) -> dict[str, str]:
    trace_id = _normalize_request_id(str(current_info.get("trace_id") or ""))
    request_id = _normalize_request_id(str(current_info.get("request_id") or ""))
    headers = {
        "x-request-id": trace_id,
        "x-caller-app": "uni-api-ember",
        "x-uni-api-ember-request-id": request_id,
        "x-caller-request-id": request_id,
    }
    trace = current_info.get("trace") if isinstance(current_info, dict) else None
    span_id = getattr(trace, "span_id", None) or current_info.get("span_id")
    trace_flags = getattr(trace, "trace_flags", None) or current_info.get("trace_flags")
    traceparent = _format_traceparent(trace_id, span_id, trace_flags)
    if traceparent:
        headers["traceparent"] = traceparent
    tracestate = str(current_info.get("tracestate") or "").strip()
    if tracestate:
        headers["tracestate"] = tracestate[:512]
    return headers


def _add_trace_headers(headers: dict[str, Any], current_info: dict[str, Any]) -> None:
    headers.update(_trace_headers_for_upstream(current_info))


def _mark_first_byte_observed(current_info: dict[str, Any]) -> None:
    if current_info.get("_first_byte_observed"):
        return
    current_info["_first_byte_observed"] = True
    trace = current_info.get("trace")
    if isinstance(trace, RequestTrace):
        trace.mark("upstream_first_chunk")
        current_info["timing_spans"] = trace.snapshot()
    runtime_gauges.end_waiting_first_byte(current_info)


async def _mark_first_byte_on_stream(generator: AsyncIterator[Any], current_info: dict[str, Any], *, skip_keepalive: bool = False):
    try:
        async with aclosing(generator):
            async for chunk in generator:
                if skip_keepalive and isinstance(chunk, str) and chunk.startswith(": keepalive"):
                    yield chunk
                    continue
                _mark_first_byte_observed(current_info)
                yield chunk
    finally:
        if current_info.get("_waiting_first_byte_active") and not current_info.get("_first_byte_observed"):
            runtime_gauges.end_waiting_first_byte(current_info)


def _message_role_summary(parsed_body: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(parsed_body, dict):
        return None, None

    roles: list[str] = []

    def append_role(value: Any) -> None:
        role = str(value or "").strip()
        if role and len(role) <= 64:
            roles.append(role)

    messages = parsed_body.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict):
                append_role(item.get("role"))

    inputs = parsed_body.get("input")
    if isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, dict):
                append_role(item.get("role"))

    if not roles:
        return None, None
    counts: dict[str, int] = defaultdict(int)
    for role in roles:
        counts[role] += 1
    ordered_counts = ",".join(f"{role}:{counts[role]}" for role in sorted(counts))
    return "/".join(roles[:16]), ordered_counts[:256]


def _record_plan_observability(current_info: dict[str, Any], plan: RoutingPlan) -> None:
    if not isinstance(current_info, dict):
        return
    current_info["role"] = plan.role
    current_info["planned_retry_count"] = max(0, int(plan.retry_count or 0))
    current_info["matching_provider_count"] = max(0, int(plan.num_matching_providers or 0))


def _record_retry_observability(attempt: Any, status_code: int, error_message: Any) -> None:
    info = request_info.get()
    if not isinstance(info, dict):
        return
    retry_count = int(info.get("retry_count") or 0) + 1
    info["retry_count"] = retry_count
    info["error_type"] = type(error_message).__name__ if not isinstance(error_message, str) else "upstream_retry"
    trace = info.get("trace")
    if isinstance(trace, RequestTrace):
        trace.mark("retry_started")
        trace.add_ms("retry_count", retry_count)
        trace.add_ms("retry_status_code", status_code)
        trace.set_tag("retry_provider", getattr(attempt, "provider_name", None))
        trace.set_tag("retry_error_type", info.get("error_type"))
        info["timing_spans"] = trace.snapshot()


def _record_cooldown_observability(attempt: Any, status_code: int, error_message: Any) -> None:
    _ = error_message
    info = request_info.get()
    if not isinstance(info, dict):
        return
    cooldown_count = int(info.get("cooldown_count") or 0) + 1
    info["cooldown_count"] = cooldown_count
    trace = info.get("trace")
    if isinstance(trace, RequestTrace):
        trace.add_ms("cooldown_count", cooldown_count)
        trace.add_ms("cooldown_status_code", status_code)
        trace.set_tag("cooldown_provider", getattr(attempt, "provider_name", None))
        info["timing_spans"] = trace.snapshot()


def _emit_request_observability(current_info: dict[str, Any]) -> None:
    if not isinstance(current_info, dict) or current_info.get("_fugue_observability_emitted"):
        return
    current_info["_fugue_observability_emitted"] = True
    try:
        emit_uni_api_ember_request_observability(
            current_info=current_info,
            runtime_metrics=runtime_gauges.snapshot(),
        )
    except Exception:
        logger.exception("Failed to enqueue Fugue request observability event")


def _debug_json_body(body: Any) -> str:
    try:
        return json.dumps(body, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return repr(body)

def _debug_header_pairs(headers: Any) -> list[dict[str, str]]:
    if not headers:
        return []

    raw_headers = getattr(headers, "raw", None)
    if raw_headers:
        pairs = []
        for key, value in raw_headers:
            if isinstance(key, bytes):
                key = key.decode("latin-1", errors="replace")
            if isinstance(value, bytes):
                value = value.decode("latin-1", errors="replace")
            pairs.append({"name": str(key), "value": str(value)})
        return pairs

    if hasattr(headers, "multi_items"):
        return [
            {"name": str(key), "value": str(value)}
            for key, value in headers.multi_items()
        ]

    if hasattr(headers, "items"):
        return [
            {"name": str(key), "value": str(value)}
            for key, value in headers.items()
        ]

    return [{"name": "<headers>", "value": str(headers)}]

def _log_debug_request_body(label: str, body: Any, **metadata: Any) -> None:
    if not is_debug:
        return
    meta = " ".join(
        f"{key}={value}"
        for key, value in metadata.items()
        if value is not None
    )
    prefix = f"{label} {meta}".rstrip()
    logger.info("%s:\n%s", prefix, _debug_json_body(body))

def _log_debug_request_headers(label: str, headers: Any, **metadata: Any) -> None:
    _log_debug_request_body(label, _debug_header_pairs(headers), **metadata)

# 从 pyproject.toml 读取版本号
try:
    with open('pyproject.toml', 'rb') as f:
        data = tomllib.load(f)
        VERSION = data['project']['version']
except Exception:
    VERSION = 'unknown'
logger.info("VERSION: %s", VERSION)

PUBLIC_HEALTH_PATHS = {"/healthz"}


def _is_public_health_request(request: Request) -> bool:
    return request.method in {"GET", "HEAD"} and request.url.path in PUBLIC_HEALTH_PATHS


async def create_tables():
    if DISABLE_DATABASE:
        return
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # 检查并添加缺失的列 - 扩展此简易迁移以支持 SQLite 和 PostgreSQL
        db_type = os.getenv("DB_TYPE", "sqlite").lower()
        if db_type in ["sqlite", "postgres"]:
            def check_and_add_columns(connection):
                inspector = inspect(connection)
                for table in [RequestStat, ChannelStat]:
                    table_name = table.__tablename__
                    existing_columns = {col['name'] for col in inspector.get_columns(table_name)}

                    for column_name, column in table.__table__.columns.items():
                        if column_name not in existing_columns:
                            # 适配 PostgreSQL 和 SQLite 的类型映射
                            col_type = column.type.compile(connection.dialect)
                            default = _get_default_sql(column.default) if db_type == "sqlite" else "" # PostgreSQL 的默认值处理更复杂，暂不处理

                            # 使用标准的 ALTER TABLE 语法
                            connection.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {col_type}{default}'))
                            logger.info(f"Added column '{column_name}' to table '{table_name}'.")

            await conn.run_sync(check_and_add_columns)

def _map_sa_type_to_sql_type(sa_type):
    type_map = {
        sqltypes.Integer: "INTEGER",
        sqltypes.String: "TEXT",
        sqltypes.Float: "REAL",
        sqltypes.Boolean: "BOOLEAN",
        sqltypes.DateTime: "DATETIME",
        sqltypes.Text: "TEXT"
    }
    return type_map.get(type(sa_type), "TEXT")

def _get_default_sql(default):
    if default is None:
        return ""
    if isinstance(default.arg, bool):
        return f" DEFAULT {str(default.arg).upper()}"
    if isinstance(default.arg, (int, float)):
        return f" DEFAULT {default.arg}"
    if isinstance(default.arg, str):
        return f" DEFAULT '{default.arg}'"
    return ""

def init_preference(all_config, preference_key, default_timeout=DEFAULT_TIMEOUT):
    # 存储超时配置
    preference_dict = {}
    preferences = safe_get(all_config, "preferences", default={})
    providers = safe_get(all_config, "providers", default=[])
    if preferences:
        if isinstance(preferences.get(preference_key), int):
            preference_dict["default"] = preferences.get(preference_key)
        else:
            for model_name, timeout_value in preferences.get(preference_key, {"default": default_timeout}).items():
                preference_dict[model_name] = timeout_value
            if "default" not in preferences.get(preference_key, {}):
                preference_dict["default"] = default_timeout

    result = defaultdict(lambda: defaultdict(lambda: default_timeout))
    for provider in providers:
        provider_preference_settings = safe_get(provider, "preferences", preference_key, default={})
        if provider_preference_settings:
            for model_name, timeout_value in provider_preference_settings.items():
                result[provider['provider']][model_name] = timeout_value

    result["global"] = preference_dict
    # print("result", json.dumps(result, indent=4))

    return result


def _build_user_api_keys_rate_limit(config: dict, api_list: list[str]) -> defaultdict:
    user_api_keys_rate_limit = defaultdict(ThreadSafeCircularList)
    for api_index, api_key in enumerate(api_list):
        user_api_keys_rate_limit[api_key] = ThreadSafeCircularList(
            [api_key],
            safe_get(config, "api_keys", api_index, "preferences", "rate_limit", default={"default": "999999/min"}),
            "round_robin",
        )
    return user_api_keys_rate_limit


def _build_admin_api_keys(api_keys_db: list[dict]) -> list[str]:
    admin_api_key = []
    for item in api_keys_db:
        if "admin" in item.get("role", ""):
            admin_api_key.append(item.get("api"))
    if admin_api_key:
        return admin_api_key
    if api_keys_db:
        return [api_keys_db[0].get("api")]

    from utils import yaml_error_message

    if yaml_error_message:
        raise HTTPException(
            status_code=500,
            detail={"error": yaml_error_message},
        )
    raise HTTPException(
        status_code=500,
        detail={"error": "No API key found in api.yaml"},
    )


async def refresh_runtime_state(app: FastAPI) -> None:
    config = getattr(app.state, "config", {}) or {}
    api_keys_db = getattr(app.state, "api_keys_db", []) or []
    api_list = getattr(app.state, "api_list", []) or []

    app.state.user_api_keys_rate_limit = _build_user_api_keys_rate_limit(config, api_list)
    app.state.global_rate_limit = parse_rate_limit(
        safe_get(config, "preferences", "rate_limit", default="999999/min")
    )
    app.state.admin_api_key = _build_admin_api_keys(api_keys_db)
    app.state.provider_timeouts = init_preference(config, "model_timeout", DEFAULT_TIMEOUT)
    app.state.keepalive_interval = init_preference(config, "keepalive_interval", 99999)
    app.state.models_list = build_api_key_models_map(config, api_list)
    app.state.routing_index = build_routing_index(config, api_list)
    app.state.model_response_cache = build_api_key_model_response_cache(api_list, app.state.models_list)

    if not DISABLE_DATABASE:
        app.state.paid_api_keys_states = {}
        for paid_key in api_list:
            await update_paid_api_keys_states(app, paid_key)


def get_runtime_api_list() -> list[str]:
    runtime_api_list = getattr(app.state, "api_list", None)
    if runtime_api_list:
        return runtime_api_list
    config = getattr(app.state, "config", {}) or {}
    return [item.get("api") for item in config.get("api_keys", []) if item.get("api")]

def get_current_model_prices(model_name: str):
    """
    根据当前配置偏好，返回指定模型的 prompt_price 和 completion_price（单位：$/M tokens）
    """
    try:
        model_price = safe_get(app.state.config, 'preferences', "model_price", default={})
        price_str = next((model_price[k] for k in model_price.keys() if model_name and model_name.startswith(k)), model_price.get("default", "0.3,1"))
        parts = [p.strip() for p in str(price_str).split(",")]
        prompt_price = float(parts[0]) if len(parts) > 0 and parts[0] != "" else 0.3
        completion_price = float(parts[1]) if len(parts) > 1 and parts[1] != "" else 1.0
        return prompt_price, completion_price
    except Exception:
        return 0.3, 1.0

async def compute_total_cost_from_db(filter_api_key: Optional[str] = None, start_dt_obj: Optional[datetime] = None) -> float:
    """
    直接从数据库历史记录累计成本：
    sum((prompt_tokens*prompt_price + completion_tokens*completion_price)/1e6)
    """
    if DISABLE_DATABASE:
        return 0.0
    async with async_session() as session:
        expr = (func.coalesce(RequestStat.prompt_tokens, 0) * func.coalesce(RequestStat.prompt_price, 0.3) + func.coalesce(RequestStat.completion_tokens, 0) * func.coalesce(RequestStat.completion_price, 1.0)) / 1000000.0
        query = select(func.coalesce(func.sum(expr), 0.0))
        if filter_api_key:
            query = query.where(RequestStat.api_key == filter_api_key)
        if start_dt_obj:
            query = query.where(RequestStat.timestamp >= start_dt_obj)
        result = await session.execute(query)
        total_cost = result.scalar_one() or 0.0
        try:
            total_cost = float(total_cost)
        except Exception:
            total_cost = 0.0
        return total_cost

async def update_paid_api_keys_states(app, paid_key):
    """
    更新付费API密钥的状态

    参数:
    app - FastAPI应用实例
    check_index - API密钥在配置中的索引
    paid_key - 需要更新状态的API密钥
    """
    try:
        check_index = app.state.api_list.index(paid_key)
    except Exception:
        raise HTTPException(
            status_code=403,
            detail={"error": "Invalid or missing API Key"}
        )
    credits = safe_get(app.state.config, 'api_keys', check_index, "preferences", "credits", default=-1)
    created_at = safe_get(app.state.config, 'api_keys', check_index, "preferences", "created_at", default=datetime.now(timezone.utc) - timedelta(days=30))
    created_at = created_at.astimezone(timezone.utc)

    # 关键修改：总消耗改为从历史数据逐条累计当时价格
    total_cost = await compute_total_cost_from_db(filter_api_key=paid_key, start_dt_obj=created_at)

    if credits != -1:
        # 仍返回聚合的 token 统计，供前端展示
        all_tokens_info = await get_usage_data(filter_api_key=paid_key, start_dt_obj=created_at)

        app.state.paid_api_keys_states[paid_key] = {
            "credits": credits,
            "created_at": created_at,
            "all_tokens_info": all_tokens_info,
            "total_cost": total_cost,
            "enabled": True if total_cost <= credits else False
        }

    return credits, total_cost
        # logger.info(f"app.state.paid_api_keys_states {paid_key}: {json.dumps({k: v.isoformat() if k == 'created_at' else v for k, v in app.state.paid_api_keys_states[paid_key].items()}, indent=4)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码
    if not DISABLE_DATABASE:
        await create_tables()

    if app and not hasattr(app.state, 'config'):
        # logger.warning("Config not found, attempting to reload")
        app.state.config, app.state.api_keys_db, app.state.api_list = await load_config(app)
        # from ruamel.yaml.timestamp import TimeStamp
        # def json_default(obj):
        #     if isinstance(obj, TimeStamp):
        #         return obj.isoformat()
        #     raise TypeError
        # print("app.state.config", json.dumps(app.state.config, indent=4, ensure_ascii=False, default=json_default))

        await refresh_runtime_state(app)

    if app and not hasattr(app.state, 'client_manager'):

        default_config = {
            "headers": {
                "User-Agent": "curl/7.68.0",
                "Accept": "*/*",
                "Accept-Encoding": "identity",
            },
            "http2": True,
            "verify": True,
            "follow_redirects": True
        }

        # 初始化客户端管理器
        app.state.client_manager = ClientManager(pool_size=100)
        await app.state.client_manager.init(default_config)


    if app and not hasattr(app.state, "channel_manager"):
        if app.state.config and 'preferences' in app.state.config:
            COOLDOWN_PERIOD = app.state.config['preferences'].get('cooldown_period', 300)
        else:
            COOLDOWN_PERIOD = 300

        app.state.channel_manager = ChannelManager(cooldown_period=COOLDOWN_PERIOD)

    if app and not hasattr(app.state, "error_triggers"):
        if app.state.config and 'preferences' in app.state.config:
            ERROR_TRIGGERS = app.state.config['preferences'].get('error_triggers', [])
        else:
            ERROR_TRIGGERS = []
        app.state.error_triggers = ERROR_TRIGGERS

    await start_fugue_observability_from_env(service_version=VERSION)

    yield
    # 关闭时的代码
    # await app.state.client.aclose()
    await stop_fugue_observability()
    if hasattr(app.state, 'client_manager'):
        await app.state.client_manager.close()

app = FastAPI(lifespan=lifespan, debug=is_debug)

def generate_markdown_docs():
    openapi_schema = app.openapi()

    markdown = f"# {openapi_schema['info']['title']}\n\n"
    markdown += f"Version: {openapi_schema['info']['version']}\n\n"
    markdown += f"{openapi_schema['info'].get('description', '')}\n\n"

    markdown += "## API Endpoints\n\n"

    paths = openapi_schema['paths']
    for path, path_info in paths.items():
        for method, operation in path_info.items():
            markdown += f"### {method.upper()} {path}\n\n"
            markdown += f"{operation.get('summary', '')}\n\n"
            markdown += f"{operation.get('description', '')}\n\n"

            if 'parameters' in operation:
                markdown += "Parameters:\n"
                for param in operation['parameters']:
                    markdown += f"- {param['name']} ({param['in']}): {param.get('description', '')}\n"

            markdown += "\n---\n\n"

    return markdown

@app.get("/docs/markdown")
async def get_markdown_docs():
    markdown = generate_markdown_docs()
    return Response(
        content=markdown,
        media_type="text/markdown"
    )

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     error_messages = []
#     for error in exc.errors():
#         # 将字段路径转换为点分隔格式（例如 body.model -> model）
#         field = ".".join(str(loc) for loc in error["loc"] if loc not in ("body", "query", "path"))
#         error_type = error["type"]

#         # 生成更友好的错误消息
#         if error_type == "value_error.missing":
#             msg = f"字段 '{field}' 是必填项"
#         elif error_type == "type_error.integer":
#             msg = f"字段 '{field}' 必须是整数类型"
#         elif error_type == "type_error.str":
#             msg = f"字段 '{field}' 必须是字符串类型"
#         else:
#             msg = error["msg"]

#         error_messages.append({
#             "field": field,
#             "message": msg,
#             "type": error_type
#         })

#     return JSONResponse(
#         status_code=422,
#         content={"detail": error_messages},
#     )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 404:
        token = await get_api_key(request)
        logger.error(f"404 Error: {exc.detail} api_key: {_mask_secret_for_log(token)}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

request_info = contextvars.ContextVar('request_info', default={})

async def parse_request_body(request: Request):
    if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
        try:
            body_bytes = await request.body()
            if not body_bytes:
                return None
            return await asyncio.to_thread(json.loads, body_bytes)
        except json.JSONDecodeError:
            return None
    return None

def _messages_request_last_text(parsed_body: Any) -> Optional[str]:
    if not isinstance(parsed_body, dict):
        return None

    messages = parsed_body.get("messages")
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
        if not isinstance(content, list):
            continue
        for part in reversed(content):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                return text
            nested_content = part.get("content")
            if isinstance(nested_content, str) and nested_content:
                return nested_content
    return None

class ChannelManager:
    def __init__(self, cooldown_period=300):
        self._excluded_models = defaultdict(lambda: None)
        self.cooldown_period = cooldown_period

    async def exclude_model(self, provider: str, model: str):
        model_key = f"{provider}/{model}"
        self._excluded_models[model_key] = datetime.now()

    async def is_model_excluded(self, provider: str, model: str, cooldown_period=0) -> bool:
        model_key = f"{provider}/{model}"
        excluded_time = self._excluded_models[model_key]
        if not excluded_time:
            return False

        if datetime.now() - excluded_time > timedelta(seconds=cooldown_period):
            del self._excluded_models[model_key]
            return False
        return True

    async def get_available_providers(self, providers: list) -> list:
        """过滤出可用的providers，仅排除不可用的模型"""
        available_providers = []
        for provider in providers:
            provider_name = provider['provider']
            model_dict = provider['model'][0]  # 获取唯一的模型字典
            # source_model = list(model_dict.keys())[0]  # 源模型名称
            target_model = list(model_dict.values())[0]  # 目标模型名称
            cooldown_period = provider.get('preferences', {}).get('cooldown_period', self.cooldown_period)

            # 检查该模型是否被排除
            if not await self.is_model_excluded(provider_name, target_model, cooldown_period):
                available_providers.append(provider)

        return available_providers

# 根据数据库类型，动态创建信号量
# SQLite 需要严格的串行写入，而 PostgreSQL 可以处理高并发
if os.getenv("DB_TYPE", "sqlite").lower() == 'sqlite':
    db_semaphore = Semaphore(1)
    logger.info("Database semaphore configured for SQLite (1 concurrent writer).")
else: # For postgres
    # 允许50个并发写入操作，这对于PostgreSQL来说是合理的
    db_semaphore = Semaphore(50)
    logger.info("Database semaphore configured for PostgreSQL (50 concurrent writers).")

async def update_stats(current_info):
    if DISABLE_DATABASE:
        return

    # 在成功请求时，快照当前价格，写入数据库
    try:
        if current_info.get("success") and current_info.get("model"):
            prompt_price, completion_price = get_current_model_prices(current_info["model"])
            current_info["prompt_price"] = prompt_price
            current_info["completion_price"] = completion_price
    except Exception:
        pass

    try:
        # 等待获取数据库访问权限
        async with db_semaphore:
            async with async_session() as session:
                async with session.begin():
                    try:
                        columns = [column.key for column in RequestStat.__table__.columns]
                        filtered_info = {k: v for k, v in current_info.items() if k in columns}

                        # 清洗字符串中的 NUL 字符，防止 PostgreSQL 报错
                        for key, value in filtered_info.items():
                            if isinstance(value, str):
                                filtered_info[key] = value.replace('\x00', '')
                            elif key == "timing_spans" and isinstance(value, dict):
                                filtered_info[key] = json.dumps(value, ensure_ascii=False, default=str)

                        new_request_stat = RequestStat(**filtered_info)
                        session.add(new_request_stat)
                        await session.commit()
                    except Exception as e:
                        await session.rollback()
                        logger.error(f"Error updating stats: {str(e)}")
                        if is_debug:
                            import traceback
                            traceback.print_exc()

        check_key = current_info["api_key"]
        if check_key and check_key in app.state.paid_api_keys_states and current_info["total_tokens"] > 0:
            await update_paid_api_keys_states(app, check_key)
    except Exception as e:
        logger.error(f"Error acquiring database lock: {str(e)}")
        if is_debug:
            import traceback
            traceback.print_exc()

async def update_channel_stats(request_id, provider, model, api_key, success, provider_api_key: str = None):
    if DISABLE_DATABASE:
        return

    try:
        async with db_semaphore:
            async with async_session() as session:
                async with session.begin():
                    try:
                        channel_stat = ChannelStat(
                            request_id=request_id,
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            provider_api_key=provider_api_key,
                            success=success,
                        )
                        session.add(channel_stat)
                        await session.commit()
                    except Exception as e:
                        await session.rollback()
                        logger.error(f"Error updating channel stats: {str(e)}")
                        if is_debug:
                            import traceback
                            traceback.print_exc()
    except Exception as e:
        logger.error(f"Error acquiring database lock: {str(e)}")
        if is_debug:
            import traceback
            traceback.print_exc()

class LoggingStreamingResponse(Response):
    def __init__(self, content, status_code=200, headers=None, media_type=None, current_info=None):
        super().__init__(content=None, status_code=status_code, headers=headers, media_type=media_type)
        self.body_iterator = content
        self._closed = False
        self.current_info = current_info
        self._sse_buffer = ""

        # Remove Content-Length header if it exists
        if 'content-length' in self.headers:
            del self.headers['content-length']
        # Set Transfer-Encoding to chunked
        self.headers['transfer-encoding'] = 'chunked'

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        trace = self.current_info.get("trace") if isinstance(self.current_info, dict) else None
        self.current_info["status_code"] = self.status_code
        if isinstance(trace, RequestTrace):
            trace.mark("downstream_response_start")
            self.current_info["timing_spans"] = trace.snapshot()
        await send({
            'type': 'http.response.start',
            'status': self.status_code,
            'headers': self.raw_headers,
        })

        try:
            async for chunk in self._logging_iterator():
                await send({
                    'type': 'http.response.body',
                    'body': chunk,
                    'more_body': True,
                })
        except Exception as e:
            # 记录异常但不重新抛出，避免"Task exception was never retrieved"
            logger.error(f"Error in streaming response: {type(e).__name__}: {str(e)}")
            if is_debug:
                import traceback
                traceback.print_exc()
            # 发送错误消息给客户端（如果可能）
            try:
                error_data = json.dumps({"error": f"Streaming error: {str(e)}"})
                await send({
                    'type': 'http.response.body',
                    'body': f"data: {error_data}\n\n".encode('utf-8'),
                    'more_body': True,
                })
            except Exception as e:
                logger.error(f"Error sending error message: {str(e)}")
        finally:
            if hasattr(self.body_iterator, 'aclose') and not self._closed:
                await _call_cleanup_safely(
                    self.body_iterator.aclose,
                    label="Downstream streaming body iterator",
                )
                self._closed = True

            final_send_cancelled: asyncio.CancelledError | None = None
            try:
                await send({
                    'type': 'http.response.body',
                    'body': b'',
                    'more_body': False,
                })
            except asyncio.CancelledError as exc:
                final_send_cancelled = exc
            except Exception as exc:
                logger.warning(
                    "Error sending final streaming response body",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

            process_time = time() - self.current_info["start_time"]
            self.current_info["process_time"] = process_time
            if isinstance(trace, RequestTrace):
                trace.mark("stream_end")
                trace.mark("usage_recorded")
                self.current_info["timing_spans"] = trace.snapshot()
                logger.info(
                    "trace_span trace_id=%s request_id=%s endpoint=%s spans=%s",
                    self.current_info.get("trace_id"),
                    self.current_info.get("request_id"),
                    self.current_info.get("endpoint"),
                    self.current_info.get("timing_spans"),
                )
            _emit_request_observability(self.current_info)
            await update_stats(self.current_info)
            if final_send_cancelled is not None:
                raise final_send_cancelled

    async def _logging_iterator(self):
        async for chunk in self.body_iterator:
            _mark_first_byte_observed(self.current_info)
            if isinstance(chunk, str):
                chunk = chunk.encode('utf-8')
            if self.current_info.get("endpoint").endswith("/v1/audio/speech"):
                yield chunk
                continue

            try:
                text = chunk.decode("utf-8", errors="replace")
            except Exception:
                yield chunk
                continue

            if is_debug:
                try:
                    logger.info(text.encode("utf-8").decode("unicode_escape"))
                except Exception:
                    logger.info(text)

            # Stream may contain multiple SSE lines per chunk, and/or partial lines.
            self._sse_buffer += text
            while "\n" in self._sse_buffer:
                line, self._sse_buffer = self._sse_buffer.split("\n", 1)
                line = line.rstrip("\r")
                if not line or line.startswith(":") or line.startswith("event:"):
                    continue

                data = None
                if line.startswith("data:"):
                    data = line.removeprefix("data:").lstrip()
                elif line.startswith("{") or line.startswith("["):
                    data = line

                if not data:
                    continue
                if data.startswith("[DONE]") or data.startswith("OK"):
                    continue

                # Avoid parsing every delta event; only parse when usage is present.
                if "\"usage\"" not in data:
                    continue

                try:
                    resp = await asyncio.to_thread(json.loads, data)
                except Exception:
                    continue

                usage_obj = None
                if isinstance(resp, dict):
                    usage_obj = resp.get("usage") or safe_get(resp, "response", "usage", default=None) or safe_get(resp, "message", "usage", default=None)
                if not isinstance(usage_obj, dict):
                    continue

                prompt_tokens = usage_obj.get("prompt_tokens")
                completion_tokens = usage_obj.get("completion_tokens")
                if prompt_tokens is None and "input_tokens" in usage_obj:
                    prompt_tokens = usage_obj.get("input_tokens")
                if completion_tokens is None and "output_tokens" in usage_obj:
                    completion_tokens = usage_obj.get("output_tokens")

                try:
                    prompt_tokens = int(prompt_tokens or 0)
                except Exception:
                    prompt_tokens = 0
                try:
                    completion_tokens = int(completion_tokens or 0)
                except Exception:
                    completion_tokens = 0

                total_tokens = usage_obj.get("total_tokens")
                try:
                    total_tokens = int(total_tokens) if total_tokens is not None else (prompt_tokens + completion_tokens)
                except Exception:
                    total_tokens = prompt_tokens + completion_tokens

                self.current_info["prompt_tokens"] = prompt_tokens
                self.current_info["completion_tokens"] = completion_tokens
                self.current_info["total_tokens"] = total_tokens
            yield chunk

    async def close(self):
        if not self._closed:
            self._closed = True
            if hasattr(self.body_iterator, 'aclose'):
                await _call_cleanup_safely(
                    self.body_iterator.aclose,
                    label="Downstream streaming body iterator",
                )

async def get_api_key(request: Request):
    token = None
    if request.headers.get("x-api-key"):
        token = request.headers.get("x-api-key")
    elif request.headers.get("Authorization"):
        api_split_list = request.headers.get("Authorization").split(" ")
        if len(api_split_list) > 1:
            token = api_split_list[1]
    return token

def get_client_ip(request: Request) -> str:
    """
    获取客户端真实 IP 地址，支持代理场景
    优先级：X-Forwarded-For > X-Real-IP > CF-Connecting-IP > True-Client-IP > request.client.host
    """
    # 1. X-Forwarded-For: 最常用的代理头，格式为 "client, proxy1, proxy2"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # 取第一个 IP（真实客户端 IP）
        return forwarded_for.split(",")[0].strip()

    # 2. X-Real-IP: nginx 常用
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # 3. CF-Connecting-IP: Cloudflare 使用
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # 4. True-Client-IP: 部分 CDN 使用
    true_client_ip = request.headers.get("True-Client-IP")
    if true_client_ip:
        return true_client_ip.strip()

    # 5. 回退到直连 IP
    return request.client.host if request.client else "unknown"

async def monitor_disconnect(request: Request, disconnect_event: asyncio.Event) -> None:
    try:
        while not disconnect_event.is_set():
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                disconnect_event.set()
                return
    except asyncio.CancelledError:
        return
    except Exception:
        disconnect_event.set()

class StatsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # 如果是 OPTIONS 请求，直接放行，由 CORSMiddleware 处理
        if request.method == "OPTIONS":
            return await call_next(request)
        if _is_public_health_request(request):
            return await call_next(request)

        start_time = time()
        incoming_trace = _incoming_trace_context(request.headers)
        trace = RequestTrace(
            trace_id=incoming_trace["trace_id"],
            parent_span_id=incoming_trace.get("parent_span_id"),
            trace_flags=incoming_trace.get("trace_flags"),
            tracestate=incoming_trace.get("tracestate"),
        )
        if incoming_trace.get("x_request_id"):
            trace.set_tag("x_request_id", incoming_trace.get("x_request_id"))
        trace.mark("request_received")
        runtime_gauges.begin_inflight()
        await runtime_gauges.record_event_loop_lag()

        # 根据token决定是否启用道德审查
        token = await get_api_key(request)
        if not token:
            runtime_gauges.end_inflight()
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid or missing API Key"},
                headers={"x-request-id": trace.trace_id},
            )

        enable_moderation = False  # 默认不开启道德审查
        config = app.state.config

        try:
            api_list = app.state.api_list
            api_index = api_list.index(token)
        except ValueError:
            # 如果 token 不在 api_list 中，检查是否以 api_list 中的任何一个开头
            # api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
            api_index = None
            # token不在api_list中，使用默认值（不开启）

        if api_index is not None:
            enable_moderation = safe_get(config, 'api_keys', api_index, "preferences", "ENABLE_MODERATION", default=False)
            if not DISABLE_DATABASE:
                check_api_key = safe_get(config, 'api_keys', api_index, "api")
                # print("check_api_key", check_api_key)
                # logger.info(f"app.state.paid_api_keys_states {check_api_key}: {json.dumps({k: v.isoformat() if k == 'created_at' else v for k, v in app.state.paid_api_keys_states[check_api_key].items()}, indent=4)}")
                # print("app.state.paid_api_keys_states", safe_get(app.state.paid_api_keys_states, check_api_key, "enabled", default=None))
                if safe_get(app.state.paid_api_keys_states, check_api_key, "enabled", default=None) is False and \
                    not request.url.path.startswith("/v1/token_usage"):
                    runtime_gauges.end_inflight()
                    return JSONResponse(
                        status_code=429,
                        content={"error": "Balance is insufficient, please check your account."},
                        headers={"x-request-id": trace.trace_id},
                    )
        else:
            runtime_gauges.end_inflight()
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid or missing API Key"},
                headers={"x-request-id": trace.trace_id},
            )
        trace.mark("auth_done")

        # 在 app.state 中存储此请求的信息
        request_id = str(uuid.uuid4())

        # 初始化请求信息
        request_info_data = {
            "request_id": request_id,
            "trace_id": trace.trace_id,
            "span_id": trace.span_id,
            "parent_span_id": trace.parent_span_id,
            "trace_flags": trace.trace_flags,
            "tracestate": trace.tracestate,
            "x_request_id": incoming_trace.get("x_request_id"),
            "trace": trace,
            "start_time": start_time,
            "endpoint": f"{request.method} {request.url.path}",
            "client_ip": get_client_ip(request),
            "process_time": 0,
            "first_response_time": -1,
            "provider": None,
            "model": None,
            "success": False,
            "api_key": token,
            "is_flagged": False,
            "text": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "timing_spans": trace.snapshot(),
        }

        # 设置请求信息到上下文
        current_request_info = request_info.set(request_info_data)
        current_info = request_info.get()
        disconnect_event: Optional[asyncio.Event] = None
        disconnect_task: Optional[asyncio.Task] = None
        try:
            _log_debug_request_headers(
                "DEBUG client request headers",
                request.headers,
                method=request.method,
                endpoint=request.url.path,
                request_id=request_id,
            )
            parsed_body = await parse_request_body(request)
            trace.mark("body_parsed")
            if isinstance(parsed_body, dict):
                current_info["stream"] = parsed_body.get("stream")
                current_info["request_kind"] = request.url.path
                message_roles, role_counts = _message_role_summary(parsed_body)
                current_info["message_roles"] = message_roles
                current_info["role_counts"] = role_counts
            if parsed_body is not None:
                _log_debug_request_body(
                    "DEBUG client request body",
                    parsed_body,
                    method=request.method,
                    endpoint=request.url.path,
                    request_id=request_id,
                )
            if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
                disconnect_event = asyncio.Event()
                current_info["disconnect_event"] = disconnect_event
                disconnect_task = asyncio.create_task(monitor_disconnect(request, disconnect_event))
            if parsed_body and not request.url.path.startswith("/v1/api_config"):
                final_api_key = app.state.api_list[api_index]
                moderated_content = None
                if request.url.path.rstrip("/") == "/v1/messages":
                    if isinstance(parsed_body, dict):
                        model = str(parsed_body.get("model") or "").strip()
                        if model:
                            current_info["model"] = model
                            try:
                                await app.state.user_api_keys_rate_limit[final_api_key].next(model)
                            except Exception:
                                current_info["status_code"] = 429
                                current_info["error_type"] = "rate_limited"
                                return JSONResponse(
                                    status_code=429,
                                    content={"error": "Too many requests"}
                                )
                        moderated_content = _messages_request_last_text(parsed_body)
                    else:
                        moderated_content = None
                else:
                    if _is_video_or_asset_request_path(request.url.path):
                        model = _lingjing_request_model_for_openapi(
                            parsed_body if isinstance(parsed_body, dict) else None,
                            request.query_params,
                        )
                        current_info["model"] = model

                        try:
                            await app.state.user_api_keys_rate_limit[final_api_key].next(model)
                        except Exception:
                            current_info["status_code"] = 429
                            current_info["error_type"] = "rate_limited"
                            return JSONResponse(
                                status_code=429,
                                content={"error": "Too many requests"}
                            )

                        if isinstance(parsed_body, dict):
                            moderated_content = str(safe_get(parsed_body, "taskParams", "input", "prompt", default="") or "").strip()
                            if not moderated_content:
                                moderated_content = _video_prompt_from_body(parsed_body)
                    else:
                        request_model = await asyncio.to_thread(UnifiedRequest.model_validate, parsed_body)
                        request_model = request_model.data
                        model = request_model.model
                        current_info["model"] = model

                        try:
                            await app.state.user_api_keys_rate_limit[final_api_key].next(model)
                        except Exception:
                            current_info["status_code"] = 429
                            current_info["error_type"] = "rate_limited"
                            return JSONResponse(
                                status_code=429,
                                content={"error": "Too many requests"}
                            )

                        if request_model.request_type == "chat":
                            moderated_content = request_model.get_last_text_message()
                        elif request_model.request_type == "image":
                            moderated_content = request_model.prompt
                        elif request_model.request_type == "tts":
                            moderated_content = request_model.input
                        elif request_model.request_type == "moderation":
                            pass
                        elif request_model.request_type == "embedding":
                            if isinstance(request_model.input, list) and len(request_model.input) > 0 and isinstance(request_model.input[0], str):
                                moderated_content = "\n".join(request_model.input)
                            else:
                                moderated_content = request_model.input
                        elif request_model.request_type == "video":
                            moderated_content = request_model.get_last_text_message()
                        else:
                            logger.error(f"Unknown request type: {request_model.request_type}")

                if enable_moderation and moderated_content:
                    background_tasks_for_moderation = BackgroundTasks()
                    moderation_response = await self.moderate_content(moderated_content, api_index, background_tasks_for_moderation)
                    is_flagged = moderation_response.get('results', [{}])[0].get('flagged', False)

                    if is_flagged:
                        logger.error(f"Content did not pass the moral check: {moderated_content}")
                        process_time = time() - start_time
                        current_info["process_time"] = process_time
                        current_info["is_flagged"] = is_flagged
                        current_info["text"] = moderated_content  # 仅在标记时记录文本
                        current_info["status_code"] = 400
                        current_info["error_type"] = "moderation_flagged"
                        await update_stats(current_info)
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Content did not pass the moral check, please modify and try again."}
                        )

            response = await call_next(request)
            trace.mark("downstream_response_start")
            response.headers["x-request-id"] = trace.trace_id
            current_info["status_code"] = getattr(response, "status_code", 0) or 0

            if request.url.path.startswith("/v1") and not DISABLE_DATABASE:
                if isinstance(response, (FastAPIStreamingResponse, StarletteStreamingResponse)) or type(response).__name__ == '_StreamingResponse':
                    current_info["_defer_observability_until_stream_end"] = True
                    response = LoggingStreamingResponse(
                        content=response.body_iterator,
                        status_code=response.status_code,
                        media_type=response.media_type,
                        headers=response.headers,
                        current_info=current_info,
                    )
                elif hasattr(response, 'json'):
                    logger.info(f"Response: {await response.json()}")
                else:
                    logger.info(f"Response: type={type(response).__name__}, status_code={response.status_code}, headers={response.headers}")

            return response

        except HTTPException as e:
            # Let FastAPI's http_exception_handler format the response consistently.
            current_info["status_code"] = getattr(e, "status_code", 500)
            current_info["error_type"] = "http_exception"
            raise
        except ValidationError as e:
            logger.error("API key: %s, invalid request body: %s", _mask_secret_for_log(token), e.errors())
            content = await asyncio.to_thread(jsonable_encoder, {"detail": e.errors()})
            current_info["status_code"] = 422
            current_info["error_type"] = "validation_error"
            return JSONResponse(
                status_code=422,
                content=content
            )
        except Exception as e:
            if is_debug:
                import traceback
                traceback.print_exc()
            logger.error(f"Error processing request: {str(e)}")
            current_info["status_code"] = 500
            current_info["error_type"] = type(e).__name__
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal server error: {str(e)}"}
            )

        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task
            trace.mark("stream_end")
            current_info["process_time"] = time() - start_time
            current_info["timing_spans"] = trace.snapshot()
            logger.info(
                "trace_span trace_id=%s request_id=%s endpoint=%s spans=%s",
                current_info.get("trace_id"),
                current_info.get("request_id"),
                current_info.get("endpoint"),
                current_info.get("timing_spans"),
            )
            if not current_info.get("_defer_observability_until_stream_end"):
                _emit_request_observability(current_info)
            runtime_gauges.end_inflight()
            # print("current_request_info", current_request_info)
            request_info.reset(current_request_info)

    async def moderate_content(self, content, api_index, background_tasks: BackgroundTasks):
        moderation_request = ModerationRequest(input=content)

        # 直接调用 moderations 函数
        response = await moderations(moderation_request, background_tasks, api_index)

        # 读取流式响应的内容
        moderation_result = b""
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                moderation_result += chunk.encode('utf-8')
            else:
                moderation_result += chunk

        # 解码并解析 JSON
        moderation_data = json.loads(moderation_result.decode('utf-8'))

        return moderation_data

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有头部字段
)

app.add_middleware(StatsMiddleware)

@app.middleware("http")
async def ensure_config(request: Request, call_next):
    if _is_public_health_request(request):
        return await call_next(request)
    runtime_api_list = get_runtime_api_list()
    if app and app.state.api_keys_db and not hasattr(app.state, "models_list"):
        app.state.models_list = build_api_key_models_map(app.state.config, runtime_api_list)
    if app and app.state.api_keys_db and not hasattr(app.state, "routing_index"):
        app.state.routing_index = build_routing_index(app.state.config, runtime_api_list)
    if app and app.state.api_keys_db and not hasattr(app.state, "model_response_cache"):
        app.state.model_response_cache = build_api_key_model_response_cache(runtime_api_list, app.state.models_list)
    return await call_next(request)


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "ok", "version": VERSION}


@app.get("/v1/observability/runtime", include_in_schema=False)
async def observability_runtime():
    await runtime_gauges.record_event_loop_lag()
    snapshot = runtime_gauges.snapshot()
    client_manager = getattr(app.state, "client_manager", None)
    if client_manager is not None and hasattr(client_manager, "snapshot"):
        snapshot["upstream_http_clients"] = client_manager.snapshot()
    return snapshot


class ClientManager:
    def __init__(self, pool_size=100):
        self.pool_size = pool_size
        self.clients = {}  # {host_timeout_proxy: AsyncClient}
        self._client_locks = defaultdict(asyncio.Lock)
        self._maintenance_task: Optional[asyncio.Task] = None
        self._last_sweep_closed_connections = 0
        self._last_sweep_error: Optional[str] = None
        self._last_sweep_at: Optional[datetime] = None

    async def init(self, default_config):
        self.default_config = default_config
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def _maintenance_loop(self):
        while True:
            await asyncio.sleep(10)
            await self.sweep_idle_connections()

    async def sweep_idle_connections(self) -> int:
        closed = 0
        errors: list[str] = []
        for key, client in list(self.clients.items()):
            try:
                closed += await _sweep_httpx_client_idle_connections(client)
            except Exception as exc:
                errors.append(f"{key}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "Failed to sweep upstream HTTP client idle connections: key=%s",
                    key,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        self._last_sweep_closed_connections = closed
        self._last_sweep_error = "; ".join(errors)[:512] if errors else None
        self._last_sweep_at = datetime.now(timezone.utc)
        return closed

    def snapshot(self) -> dict[str, Any]:
        return {
            "client_count": len(self.clients),
            "pool_size": self.pool_size,
            "last_sweep_closed_connections": self._last_sweep_closed_connections,
            "last_sweep_at": self._last_sweep_at.isoformat() if self._last_sweep_at else None,
            "last_sweep_error": self._last_sweep_error,
        }

    @asynccontextmanager
    async def get_client(self, base_url, proxy=None, http2: Optional[bool] = None):
        trace = _current_trace()
        if trace is not None:
            trace.mark("client_pool_acquire_start")
        runtime_gauges.begin_upstream_pool(trace)
        acquire_started_at = time()
        acquired = False
        # 从base_url中提取主机名
        parsed_url = urlparse(base_url)
        host = parsed_url.netloc

        # 创建唯一的客户端键
        client_key = f"{host}"
        if proxy:
            # 对代理URL进行规范化处理
            proxy_normalized = proxy.replace('socks5h://', 'socks5://')
            client_key += f"_{proxy_normalized}"
        if http2 is not None:
            client_key += f"_http2_{int(bool(http2))}"

        if client_key not in self.clients:
            async with self._client_locks[client_key]:
                if client_key not in self.clients:
                    timeout = httpx.Timeout(
                        connect=15.0,
                        read=None,
                        write=30.0,
                        pool=self.pool_size,
                    )
                    limits = httpx.Limits(max_connections=self.pool_size)

                    client_config = {
                        **self.default_config,
                        "timeout": timeout,
                        "limits": limits,
                    }

                    client_config = get_proxy(proxy, client_config)
                    if http2 is not None:
                        client_config["http2"] = bool(http2)

                    self.clients[client_key] = httpx.AsyncClient(**client_config)

        try:
            acquired = True
            if trace is not None:
                trace.mark("client_pool_acquire_end")
                trace.add_ms("upstream_pool_wait_ms", (time() - acquire_started_at) * 1000)
            runtime_gauges.end_upstream_pool()
            yield self.clients[client_key]
        except Exception as e:
            # if client_key in self.clients and "429" not in str(e):
            #     tmp_client = self.clients[client_key]
            #     del self.clients[client_key]  # 先删除引用
            #     await tmp_client.aclose()  # 然后关闭客户端
            # 仅在客户端主动关闭等严重错误时才考虑重建，暂时只将异常抛出
            # httpx的连接池会自动处理单个连接的失败
            # logger.warning(f"Exception with client {client_key}: {type(e).__name__}: {e}")
            raise e
        finally:
            if not acquired:
                runtime_gauges.end_upstream_pool()

    async def close(self):
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

rate_limiter = InMemoryRateLimiter()

async def rate_limit_dependency():
    if await rate_limiter.is_rate_limited("global", app.state.global_rate_limit):
        raise HTTPException(status_code=429, detail="Too many requests")

def get_preference_value(provider_timeouts, original_model):
    timeout_value = None
    original_model = original_model.lower()
    if original_model in provider_timeouts:
        timeout_value = provider_timeouts[original_model]
    else:
        # 尝试模糊匹配模型
        for timeout_model in provider_timeouts:
            if timeout_model != "default" and timeout_model.lower() in original_model.lower():
                timeout_value = provider_timeouts[timeout_model]
                break
        else:
            # 如果模糊匹配失败，使用渠道的默认值
            timeout_value = provider_timeouts.get("default", None)
    return timeout_value

def get_preference(preference_config, channel_id, original_request_model, default_value):
    original_model, request_model_name = original_request_model
    provider_timeouts = safe_get(preference_config, channel_id, default=preference_config["global"])
    timeout_value = get_preference_value(provider_timeouts, request_model_name)
    if timeout_value is None:
        timeout_value = get_preference_value(provider_timeouts, original_model)
    if timeout_value is None:
        timeout_value = get_preference_value(preference_config["global"], request_model_name)
    if timeout_value is None:
        timeout_value = get_preference_value(preference_config["global"], original_model)
    if timeout_value is None:
        timeout_value = preference_config["global"].get("default", default_value)
    # print("timeout_value", channel_id, timeout_value)
    return timeout_value

def _split_codex_api_key(raw_api_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if raw_api_key is None:
        return None, None
    raw = str(raw_api_key).strip()
    if not raw:
        return None, None
    if "," not in raw:
        return None, raw
    account_id, token = raw.split(",", 1)
    account_id = account_id.strip() or None
    token = token.strip()
    if not token:
        raise ValueError("Invalid Codex API key format: expected 'account_id,refresh_token' (refresh_token missing)")
    return account_id, token

_CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
_CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_CODEX_OAUTH_REFRESH_SKEW_SECONDS = 30

# provider_api_key_raw -> {"access_token": str, "refresh_token": str, "expires_at": float|None}
_codex_oauth_cache: dict[str, dict[str, Any]] = {}
_codex_oauth_locks: dict[str, asyncio.Lock] = {}

# provider_api_key_raw -> refresh_token
# NOTE: We intentionally key by the full raw config string (usually "account_id,refresh_token") so multiple
# Codex keys sharing the same account_id but having different refresh tokens won't overwrite each other.
_CODEX_REFRESH_TOKEN_STORE_PATH = os.getenv("CODEX_REFRESH_TOKEN_STORE_PATH", "./data/codex_refresh_tokens.json")
_codex_refresh_token_store: dict[str, str] = {}
_codex_refresh_token_store_loaded = False
_codex_refresh_token_store_lock = asyncio.Lock()

async def _ensure_codex_refresh_token_store_loaded() -> None:
    global _codex_refresh_token_store_loaded
    if _codex_refresh_token_store_loaded:
        return
    async with _codex_refresh_token_store_lock:
        if _codex_refresh_token_store_loaded:
            return
        try:
            with open(_CODEX_REFRESH_TOKEN_STORE_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for k, v in payload.items():
                    key = str(k).strip()
                    val = str(v).strip()
                    if key and val:
                        _codex_refresh_token_store[key] = val
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Failed to load Codex refresh token store '%s': %s", _CODEX_REFRESH_TOKEN_STORE_PATH, e)
        _codex_refresh_token_store_loaded = True

async def _reload_codex_refresh_token_store() -> None:
    global _codex_refresh_token_store_loaded
    async with _codex_refresh_token_store_lock:
        _codex_refresh_token_store.clear()
        try:
            with open(_CODEX_REFRESH_TOKEN_STORE_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for k, v in payload.items():
                    key = str(k).strip()
                    val = str(v).strip()
                    if key and val:
                        _codex_refresh_token_store[key] = val
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Failed to reload Codex refresh token store '%s': %s", _CODEX_REFRESH_TOKEN_STORE_PATH, e)
        _codex_refresh_token_store_loaded = True

async def _get_codex_refresh_token_from_store(provider_api_key_raw: Optional[str], *, force_reload: bool = False) -> Optional[str]:
    if provider_api_key_raw is None:
        return None
    key = str(provider_api_key_raw).strip()
    if not key:
        return None
    if force_reload:
        await _reload_codex_refresh_token_store()
    else:
        await _ensure_codex_refresh_token_store_loaded()
    token = _codex_refresh_token_store.get(key)
    return str(token) if token else None

async def _persist_codex_refresh_token(provider_api_key_raw: Optional[str], refresh_token: Optional[str]) -> None:
    if provider_api_key_raw is None:
        return
    key = str(provider_api_key_raw).strip()
    rt = str(refresh_token or "").strip()
    if not key or not rt:
        return
    await _ensure_codex_refresh_token_store_loaded()

    async with _codex_refresh_token_store_lock:
        if _codex_refresh_token_store.get(key) == rt:
            return
        _codex_refresh_token_store[key] = rt
        try:
            store_dir = os.path.dirname(_CODEX_REFRESH_TOKEN_STORE_PATH)
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
            tmp_path = f"{_CODEX_REFRESH_TOKEN_STORE_PATH}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(_codex_refresh_token_store, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, _CODEX_REFRESH_TOKEN_STORE_PATH)
        except Exception as e:
            logger.warning("Failed to persist Codex refresh token store '%s': %s", _CODEX_REFRESH_TOKEN_STORE_PATH, e)

def _codex_oauth_lock(key: str) -> asyncio.Lock:
    lock = _codex_oauth_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _codex_oauth_locks[key] = lock
    return lock

def _codex_access_token_is_valid(entry: dict[str, Any]) -> bool:
    token = entry.get("access_token")
    if not token:
        return False
    expires_at = entry.get("expires_at")
    if expires_at is None:
        return True
    try:
        return time() < float(expires_at) - _CODEX_OAUTH_REFRESH_SKEW_SECONDS
    except Exception:
        return True

async def _refresh_codex_access_token(refresh_token: str, proxy: Optional[str]) -> dict[str, Any]:
    rt = (refresh_token or "").strip()
    if not rt:
        raise HTTPException(status_code=401, detail="Codex refresh_token missing")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    data = {
        "client_id": _CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": rt,
        "scope": "openid profile email",
    }

    try:
        async with app.state.client_manager.get_client(_CODEX_OAUTH_TOKEN_URL, proxy) as client:
            resp = await client.post(_CODEX_OAUTH_TOKEN_URL, data=data, headers=headers, timeout=30)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Codex token refresh request failed: {type(e).__name__}: {e}")

    if resp.status_code != 200:
        body = (resp.text or "").strip()
        raise HTTPException(status_code=401, detail=f"Codex token refresh failed: status {resp.status_code}: {body}")

    try:
        payload = resp.json()
    except Exception:
        payload = {}

    access_token = str(payload.get("access_token") or "").strip()
    if not access_token:
        raise HTTPException(status_code=401, detail=f"Codex token refresh returned empty access_token: {resp.text}")

    new_refresh_token = str(payload.get("refresh_token") or "").strip() or None
    expires_in = payload.get("expires_in")
    expires_at = None
    try:
        expires_in_int = int(expires_in)
        if expires_in_int > 0:
            expires_at = time() + expires_in_int
    except Exception:
        expires_at = None

    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "expires_at": expires_at,
    }

async def _get_codex_access_token(provider_name: str, provider_api_key_raw: str, proxy: Optional[str]) -> str:
    # provider_api_key_raw is the stable key-id we use for rate-limit/cooling/logging.
    account_id, refresh_token_from_config = _split_codex_api_key(provider_api_key_raw)
    if not refresh_token_from_config:
        raise HTTPException(status_code=401, detail="Codex refresh_token missing")

    persisted_refresh_token = await _get_codex_refresh_token_from_store(provider_api_key_raw)
    if persisted_refresh_token:
        refresh_token_from_config = persisted_refresh_token

    lock = _codex_oauth_lock(provider_api_key_raw)
    async with lock:
        entry = _codex_oauth_cache.get(provider_api_key_raw) or {}
        if _codex_access_token_is_valid(entry):
            return str(entry["access_token"])

        old_refresh_token = str(entry.get("refresh_token") or refresh_token_from_config).strip()
        try:
            refreshed = await _refresh_codex_access_token(old_refresh_token, proxy)
        except HTTPException as e:
            detail = str(getattr(e, "detail", "") or "")
            if "refresh_token_reused" in detail:
                latest = await _get_codex_refresh_token_from_store(provider_api_key_raw, force_reload=True)
                if latest and latest != old_refresh_token:
                    refreshed = await _refresh_codex_access_token(latest, proxy)
                    old_refresh_token = latest
                else:
                    raise
            raise

        updated_refresh_token = refreshed.get("refresh_token") or old_refresh_token
        _codex_oauth_cache[provider_api_key_raw] = {
            "access_token": refreshed["access_token"],
            "refresh_token": updated_refresh_token,
            "expires_at": refreshed.get("expires_at"),
        }
        await _persist_codex_refresh_token(provider_api_key_raw, updated_refresh_token)
        return str(refreshed["access_token"])


async def _resolve_codex_upstream_auth(
    provider_name: str,
    provider_api_key_raw: Optional[str],
    proxy: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if provider_api_key_raw is None:
        return None, None

    raw = str(provider_api_key_raw).strip()
    if not raw:
        return None, None

    # Support direct Codex-compatible proxies that only need a fixed Bearer token.
    if "," not in raw:
        return raw, None

    codex_account_id, _ = _split_codex_api_key(raw)
    api_key = await _get_codex_access_token(provider_name, raw, proxy)
    return api_key, codex_account_id

# 在 process_request 函数中更新成功和失败计数
async def process_request(
    request: Union[RequestModel, ImageGenerationRequest, ImageEditRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest],
    provider: Dict,
    background_tasks: BackgroundTasks,
    endpoint=None,
    role=None,
    timeout_value=DEFAULT_TIMEOUT,
    keepalive_interval=None,
    provider_api_key_raw: Optional[str] = None,
):
    timeout_value = int(timeout_value)
    model_dict = provider["_model_dict_cache"]
    original_model = model_dict[request.model]
    if provider_api_key_raw is None:
        provider_api_key_raw = await select_provider_api_key_raw(
            provider,
            original_model,
            get_runtime_api_list(),
        )

    engine, stream_mode = get_engine(provider, endpoint, original_model)

    if stream_mode is not None:
        request.stream = stream_mode

    proxy = safe_get(app.state.config, "preferences", "proxy", default=None)  # global proxy
    proxy = safe_get(provider, "preferences", "proxy", default=proxy)  # provider proxy

    api_key = provider_api_key_raw
    codex_account_id = None
    if engine == "codex" and provider_api_key_raw:
        try:
            api_key, codex_account_id = await _resolve_codex_upstream_auth(
                provider["provider"],
                provider_api_key_raw,
                proxy,
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Gemini preview TTS returns inline audio; force non-stream so we can return a single OpenAI-style JSON response.
    try:
        has_audio_modality = any(str(m).lower() == "audio" for m in (getattr(request, "modalities", None) or []))
    except Exception:
        has_audio_modality = False
    if engine in ["gemini", "vertex-gemini"] and (has_audio_modality or "preview-tts" in original_model.lower()):
        request.stream = False

    channel_id = f"{provider['provider']}"
    current_info = request_info.get()
    trace = current_info.get("trace") if isinstance(current_info, dict) else None
    if isinstance(current_info, dict):
        current_info["stream"] = bool(getattr(request, "stream", False))
        current_info["role"] = role
    if isinstance(trace, RequestTrace):
        trace.mark("provider_selected")
        trace.set_tag("provider", channel_id)
        trace.set_tag("model", request.model)
    if engine != "moderation":
        _log_stdout_request_summary(channel_id, request.model, engine, role)

    last_message_role = safe_get(request, "messages", -1, "role", default=None)
    url, headers, payload = await get_payload(request, engine, provider, api_key, endpoint=endpoint)
    if engine == "codex" and codex_account_id:
        headers.setdefault("Chatgpt-Account-Id", str(codex_account_id))
    headers.update(safe_get(provider, "preferences", "headers", default={}))  # add custom headers
    if engine == "codex":
        force_codex_client_headers(headers)
    _add_trace_headers(headers, current_info)
    if isinstance(trace, RequestTrace):
        trace.mark("provider_key_selected")

    # print("proxy", proxy)

    try:
        async with app.state.client_manager.get_client(url, proxy, http2=False if engine == "codex" else None) as client:
            downstream_stream = bool(getattr(request, "stream", None))
            force_collect_codex_stream = engine == "codex" and not downstream_stream and endpoint is None

            if downstream_stream and not force_collect_codex_stream:
                _log_debug_request_headers(
                    "DEBUG upstream request headers",
                    headers,
                    endpoint=endpoint or "/v1/chat/completions",
                    upstream_url=url,
                    provider=channel_id,
                    model=request.model,
                    actual_model=original_model,
                )
                _log_debug_request_body(
                    "DEBUG upstream request body",
                    payload,
                    endpoint=endpoint or "/v1/chat/completions",
                    upstream_url=url,
                    provider=channel_id,
                    model=request.model,
                    actual_model=original_model,
                )
                if isinstance(trace, RequestTrace):
                    trace.mark("upstream_send_start")
                runtime_gauges.begin_waiting_first_byte(current_info)
                generator = fetch_response_stream(client, url, headers, payload, engine, original_model, timeout_value)
                if isinstance(trace, RequestTrace):
                    trace.mark("upstream_headers_received")
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, True, app.state.error_triggers, keepalive_interval=keepalive_interval, last_message_role=last_message_role)
                if first_response_time == 3.1415:
                    wrapped_generator = _mark_first_byte_on_stream(wrapped_generator, current_info, skip_keepalive=True)
                else:
                    _mark_first_byte_observed(current_info)
                response = StarletteStreamingResponse(wrapped_generator, media_type="text/event-stream")
            elif force_collect_codex_stream:
                payload["stream"] = True
                headers["Accept"] = "text/event-stream"
                _log_debug_request_headers(
                    "DEBUG upstream request headers",
                    headers,
                    endpoint=endpoint or "/v1/chat/completions",
                    upstream_url=url,
                    provider=channel_id,
                    model=request.model,
                    actual_model=original_model,
                )
                _log_debug_request_body(
                    "DEBUG upstream request body",
                    payload,
                    endpoint=endpoint or "/v1/chat/completions",
                    upstream_url=url,
                    provider=channel_id,
                    model=request.model,
                    actual_model=original_model,
                )
                if isinstance(trace, RequestTrace):
                    trace.mark("upstream_send_start")
                runtime_gauges.begin_waiting_first_byte(current_info)
                generator = fetch_response_stream(client, url, headers, payload, engine, original_model, timeout_value)
                if isinstance(trace, RequestTrace):
                    trace.mark("upstream_headers_received")
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, True, app.state.error_triggers, keepalive_interval=keepalive_interval, last_message_role=last_message_role)
                if first_response_time != 3.1415:
                    _mark_first_byte_observed(current_info)
                json_data = await collect_openai_chat_completion_from_streaming_sse(wrapped_generator, model=original_model)
                _mark_first_byte_observed(current_info)
                response = StarletteStreamingResponse(iter([json_data]), media_type="application/json")
            else:
                _log_debug_request_headers(
                    "DEBUG upstream request headers",
                    headers,
                    endpoint=endpoint or "/v1/chat/completions",
                    upstream_url=url,
                    provider=channel_id,
                    model=request.model,
                    actual_model=original_model,
                )
                _log_debug_request_body(
                    "DEBUG upstream request body",
                    payload,
                    endpoint=endpoint or "/v1/chat/completions",
                    upstream_url=url,
                    provider=channel_id,
                    model=request.model,
                    actual_model=original_model,
                )
                if isinstance(trace, RequestTrace):
                    trace.mark("upstream_send_start")
                runtime_gauges.begin_waiting_first_byte(current_info)
                generator = fetch_response(client, url, headers, payload, engine, original_model, timeout_value)
                if isinstance(trace, RequestTrace):
                    trace.mark("upstream_headers_received")
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, False, app.state.error_triggers, keepalive_interval=keepalive_interval, last_message_role=last_message_role)
                _mark_first_byte_observed(current_info)

                # 处理音频和其他二进制响应
                if endpoint == "/v1/audio/speech":
                    if isinstance(wrapped_generator, bytes):
                        response = Response(content=wrapped_generator, media_type="audio/mpeg")
                else:
                    async with aclosing(wrapped_generator):
                        first_element = await anext(wrapped_generator)
                    _mark_first_byte_observed(current_info)
                    first_element = first_element.lstrip("data: ")
                    decoded_element = await asyncio.to_thread(json.loads, first_element)
                    encoded_element = await asyncio.to_thread(json.dumps, decoded_element)
                    response = StarletteStreamingResponse(iter([encoded_element]), media_type="application/json")

            # 更新成功计数和首次响应时间
            background_tasks.add_task(update_channel_stats, current_info["request_id"], channel_id, request.model, current_info["api_key"], success=True, provider_api_key=provider_api_key_raw)
            current_info["first_response_time"] = first_response_time
            current_info["success"] = True
            current_info["provider"] = channel_id
            return response

    except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError, httpx.LocalProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
        background_tasks.add_task(update_channel_stats, current_info["request_id"], channel_id, request.model, current_info["api_key"], success=False, provider_api_key=provider_api_key_raw)
        raise e

class ModelRequestHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def request_model(
        self,
        request_data: Union[RequestModel, ImageGenerationRequest, ImageEditRequest, AudioTranscriptionRequest, ModerationRequest, EmbeddingRequest],
        api_index: int,
        background_tasks: BackgroundTasks,
        endpoint=None,
    ):
        config = app.state.config
        request_model_name = request_data.model
        if not safe_get(config, 'api_keys', api_index, 'model'):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = request_info.get()
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_total_tokens = estimate_request_total_tokens(request_data)
        routing_endpoint = endpoint or "/v1/chat/completions"
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=routing_endpoint,
            request_total_tokens=request_total_tokens,
            debug=is_debug,
            provider_resolver=get_right_order_providers,
        )
        _record_plan_observability(current_info, plan)
        exclude_error_rate_limit = [
            "BrokenResourceError",
            "Proxy connection timed out",
            "Unknown error: EndOfStream",
            "'status': 'INVALID_ARGUMENT'",
            "Unable to connect to service",
            "Connection closed unexpectedly",
            "Invalid JSON payload received. Unknown name ",
            "User location is not supported for the API use",
            "The model is overloaded. Please try again later.",
            "[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] sslv3 alert handshake failure (_ssl.c:1007)",
            "<title>Worker exceeded resource limits",
        ]
        runner = UpstreamRunner(
            plan,
            endpoint=endpoint,
            debug=is_debug,
            clear_provider_auth_cache=lambda provider_api_key_raw: _codex_oauth_cache.pop(provider_api_key_raw, None),
        )
        async def before_next_attempt():
            if disconnect_event is not None and disconnect_event.is_set():
                return Response(content="", status_code=499)
            return None

        async def execute_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model

            original_request_model = (original_model, request_data.model)
            local_api_list = get_runtime_api_list()
            if provider_name.startswith("sk-") and provider_name in local_api_list:
                local_provider_api_index = local_api_list.index(provider_name)
                local_provider_scheduling_algorithm = safe_get(
                    config,
                    "api_keys",
                    local_provider_api_index,
                    "preferences",
                    "SCHEDULING_ALGORITHM",
                    default="fixed_priority",
                )
                local_provider_matching_providers = await _call_provider_resolver(
                    get_right_order_providers,
                    request_model_name,
                    config,
                    local_provider_api_index,
                    local_provider_scheduling_algorithm,
                    api_list=local_api_list,
                    models_list=app.state.models_list,
                    endpoint=routing_endpoint,
                    channel_manager=app.state.channel_manager,
                    request_total_tokens=request_total_tokens,
                    debug=is_debug,
                    routing_index=getattr(app.state, "routing_index", None),
                )
                local_timeout_value = 0
                for local_provider in local_provider_matching_providers:
                    local_provider_name = local_provider["provider"]
                    if not local_provider_name.startswith("sk-"):
                        original_request_model = (
                            local_provider["_model_dict_cache"][request_model_name],
                            request_data.model,
                        )
                        local_timeout_value += get_preference(
                            app.state.provider_timeouts,
                            local_provider_name,
                            original_request_model,
                            DEFAULT_TIMEOUT,
                        )
                local_provider_num_matching_providers = len(local_provider_matching_providers)
            else:
                local_timeout_value = get_preference(
                    app.state.provider_timeouts,
                    provider_name,
                    original_request_model,
                    DEFAULT_TIMEOUT,
                )
                local_provider_num_matching_providers = 1

            local_timeout_value = local_timeout_value * local_provider_num_matching_providers
            keepalive_interval = get_preference(
                app.state.keepalive_interval,
                provider_name,
                original_request_model,
                99999,
            )
            if keepalive_interval > local_timeout_value or provider_name.startswith("sk-"):
                keepalive_interval = None

            attempt.provider_api_key_raw = await runner.select_provider_api_key(attempt)
            process_task = asyncio.create_task(
                process_request(
                    request_data,
                    provider,
                    background_tasks,
                    endpoint,
                    plan.role,
                    local_timeout_value,
                    keepalive_interval,
                    provider_api_key_raw=attempt.provider_api_key_raw,
                )
            )
            disconnect_task: Optional[asyncio.Task] = None
            try:
                if disconnect_event is not None:
                    disconnect_task = asyncio.create_task(disconnect_event.wait())
                    done, pending = await asyncio.wait(
                        [process_task, disconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if disconnect_task in done and disconnect_event.is_set():
                        process_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await process_task
                        return Response(content="", status_code=499)

                return await process_task
            except asyncio.CancelledError:
                raise
            except Exception:
                if disconnect_event is not None and disconnect_event.is_set():
                    return Response(content="", status_code=499)
                raise
            finally:
                if disconnect_task is not None:
                    disconnect_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await disconnect_task

        def after_failure(attempt, exc, status_code, error_message):
            _ = exc
            request_model, actual_model = _log_model_names(request_data.model, attempt.original_model)
            logger.error(
                "Error %s with provider %s request_model=%s actual_model=%s API key: %s: %s",
                status_code,
                attempt.provider_name,
                request_model,
                actual_model,
                _mask_secret_for_log(attempt.provider_api_key_raw),
                error_message,
            )
            if is_debug or status_code == 500:
                import traceback

                traceback.print_exc()

        def build_final_response(completed_plan):
            current_info = request_info.get()
            if isinstance(current_info, dict):
                current_info["first_response_time"] = -1
                current_info["success"] = False
                current_info["provider"] = None
            return JSONResponse(
                status_code=completed_plan.status_code,
                content={"error": f"All {request_data.model} error: {completed_plan.error_message}"},
            )

        return await runner.run(
            execute_attempt,
            before_next_attempt=before_next_attempt,
            after_failure=after_failure,
            build_final_response=build_final_response,
            exclude_error_substrings=exclude_error_rate_limit,
            rollback_rate_limit_errors=exclude_error_rate_limit,
            allow_channel_exclusion=True,
            on_retry=_record_retry_observability,
            on_cooldown=_record_cooldown_observability,
        )

def _normalize_responses_upstream_url(base_url: str, engine: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")
    if engine != "codex":
        return base
    if base.endswith("/v1/responses") or base.endswith("/responses"):
        return base
    return f"{base}/responses"

def _normalize_responses_compact_upstream_url(base_url: str, engine: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")

    if base.endswith("/v1/responses/compact") or base.endswith("/responses/compact"):
        return base

    if engine == "codex":
        base = _normalize_responses_upstream_url(base, engine)

    if base.endswith("/v1/responses") or base.endswith("/responses"):
        return f"{base}/compact"

    if base.endswith("/compact"):
        return base

    return f"{base}/compact"

def _normalize_messages_upstream_url(base_url: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")
    if base.endswith("/v1/messages") or base.endswith("/messages"):
        return base
    return f"{base}/messages"

VIDEO_TASKS_ENDPOINT = "/v1/video/tasks"
VIDEO_ASSETS_ENDPOINT = "/v1/assets"
VIDEO_ASSET_GROUPS_ENDPOINT = "/v1/asset-groups"
CONTENT_GENERATION_TASKS_ENDPOINT = VIDEO_TASKS_ENDPOINT
LINGJING_OPENAPI_ENDPOINT_PREFIX = "/v1/openapi"
LINGJING_UPSTREAM_OPENAPI_PREFIX = "/api/entrance/openapi"
LINGJING_DEFAULT_REQUEST_MODEL = "seedance-2-0"

def _is_video_or_asset_request_path(path: str) -> bool:
    normalized = str(path or "").rstrip("/")
    return (
        normalized == VIDEO_TASKS_ENDPOINT
        or normalized.startswith(f"{VIDEO_TASKS_ENDPOINT}/")
        or normalized == VIDEO_ASSETS_ENDPOINT
        or normalized.startswith(f"{VIDEO_ASSETS_ENDPOINT}/")
        or normalized == VIDEO_ASSET_GROUPS_ENDPOINT
        or normalized.startswith(f"{VIDEO_ASSET_GROUPS_ENDPOINT}/")
    )

def _normalize_content_generation_tasks_upstream_url(base_url: str, task_id: Optional[str] = None) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")
    parsed = urlparse(base)
    path = parsed.path.rstrip("/")

    if path.endswith("/contents/generations/tasks"):
        tasks_url = base
    elif path in ("", "/"):
        tasks_url = f"{base}/api/v3/contents/generations/tasks"
    else:
        tasks_url = f"{base}/contents/generations/tasks"

    if task_id is not None:
        tasks_url = f"{tasks_url}/{quote(str(task_id), safe='')}"
    return tasks_url

def _is_lingjing_provider(provider: dict) -> bool:
    if str(provider.get("engine") or "").strip().lower() == "lingjing":
        return True
    parsed = urlparse(str(provider.get("base_url") or ""))
    return parsed.netloc.endswith("lingjingai.cn")

def _normalize_lingjing_openapi_upstream_url(base_url: str, openapi_path: str, query: str = "") -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return base

    path = "/" + str(openapi_path or "").strip("/")
    if not path.startswith("/openapi/"):
        path = "/openapi" + path

    parsed = urlparse(base)
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/api/entrance/openapi"):
        upstream_path = base_path + path[len("/openapi"):]
    elif base_path.endswith("/api/entrance"):
        upstream_path = base_path + path
    else:
        upstream_path = base_path + LINGJING_UPSTREAM_OPENAPI_PREFIX + path[len("/openapi"):]

    url = urlunparse(parsed[:2] + (upstream_path,) + ("",) * 3)
    return f"{url}?{query}" if query else url

def _lingjing_upstream_query(raw_query: str) -> str:
    pairs = parse_qsl(raw_query or "", keep_blank_values=True)
    filtered = [(key, value) for key, value in pairs if key not in {"model", "request_model"}]
    return urlencode(filtered, doseq=True)

def _normalize_lingjing_draw_task_upstream_url(base_url: str, *, method: str, task_id: Optional[str] = None) -> str:
    if method.upper() == "POST":
        return _normalize_lingjing_openapi_upstream_url(base_url, "/draw/task/submit")
    if method.upper() == "GET":
        if not task_id:
            return _normalize_lingjing_openapi_upstream_url(base_url, "/draw/task/query")
        return _normalize_lingjing_openapi_upstream_url(
            base_url,
            "/draw/task/query",
            query=f"taskId={quote(str(task_id), safe='')}",
        )
    return ""

def _parse_lingjing_credentials(provider: dict, provider_api_key_raw: Optional[str]) -> tuple[str, str]:
    access_key = str(safe_get(provider, "preferences", "access_key", default="") or "").strip()
    secret_key = str(safe_get(provider, "preferences", "secret_key", default="") or "").strip()
    raw = str(provider_api_key_raw or "").strip()

    if (not access_key or not secret_key) and raw:
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    access_key = access_key or str(parsed.get("access_key") or parsed.get("accessKey") or "").strip()
                    secret_key = secret_key or str(parsed.get("secret_key") or parsed.get("secretKey") or "").strip()
            except Exception:
                pass
        for sep in (":", ",", "|"):
            if access_key and secret_key:
                break
            if sep in raw:
                left, right = raw.split(sep, 1)
                access_key = access_key or left.strip()
                secret_key = secret_key or right.strip()

    if not access_key or not secret_key:
        raise HTTPException(status_code=400, detail="Lingjing provider requires access and secret keys")
    return access_key, secret_key

def _lingjing_headers(
    provider: dict,
    provider_api_key_raw: Optional[str],
    *,
    include_content_type: bool = False,
) -> dict[str, str]:
    access_key, secret_key = _parse_lingjing_credentials(provider, provider_api_key_raw)
    headers: dict[str, str] = {
        "X-Access-Key": access_key,
        "X-Secret-Key": secret_key,
    }
    if include_content_type:
        headers["Content-Type"] = "application/json"
    headers.update(safe_get(provider, "preferences", "headers", default={}) or {})
    return headers

def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")

def _maybe_json_object(raw: bytes) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None

def _lingjing_source_from_value(value: Any) -> dict[str, Any]:
    raw = str(value or "").strip()
    if raw.startswith("asset://"):
        return {"kind": "asset_id", "value": raw[len("asset://"):]}
    if raw.startswith("Asset-"):
        return {"kind": "asset_id", "value": raw}
    return {"kind": "url", "value": raw}

def _extract_url_from_content_part(part: dict[str, Any], type_name: str) -> str:
    value = part.get(type_name)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("url") or "").strip()
    return ""

def _video_provider_options(request_body: dict[str, Any], provider_name: str) -> dict[str, Any]:
    options = request_body.get("provider_options")
    if not isinstance(options, dict):
        return {}

    provider_options = options.get(provider_name)
    if isinstance(provider_options, dict):
        return dict(provider_options)

    common_options = {
        key: value
        for key, value in options.items()
        if not isinstance(value, dict)
    }
    return common_options

def _video_requested_provider(request_body: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(request_body, dict):
        return None
    provider = request_body.get("provider")
    if not provider and isinstance(request_body.get("route"), dict):
        provider = request_body["route"].get("provider")
    provider_name = str(provider or "").strip()
    return provider_name or None

def _video_prompt_from_body(request_body: dict[str, Any]) -> str:
    prompt = str(request_body.get("prompt") or "").strip()
    if prompt:
        return prompt

    prompt_parts: list[str] = []
    content = request_body.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and str(part.get("type") or "").strip() == "text":
                text = str(part.get("text") or "").strip()
                if text:
                    prompt_parts.append(text)
    return "\n".join(prompt_parts).strip()

def _lingjing_usage_from_role(role: Any, resource_type: str, resource_index: int) -> str:
    normalized = str(role or "").strip().lower()
    if normalized in {"first_frame", "last_frame", "reference", "keyframe", "source"}:
        return normalized
    if normalized in {"reference_image", "reference_video", "reference_audio"}:
        return "reference"
    if resource_type == "image" and resource_index == 0:
        return "first_frame"
    return "reference"

def _lingjing_resource_from_unified(resource: Any, resource_index: int) -> Optional[dict[str, Any]]:
    if not isinstance(resource, dict):
        return None

    resource_type = str(resource.get("type") or "image").strip().lower()
    if resource_type not in {"image", "video", "audio"}:
        return None

    usage = resource.get("usage", resource.get("role"))
    source = resource.get("source")
    if not isinstance(source, dict):
        value = (
            resource.get("url")
            or resource.get("asset_id")
            or resource.get("assetId")
            or resource.get("value")
        )
        source = _lingjing_source_from_value(value)

    normalized: dict[str, Any] = {
        "type": resource_type,
        "usage": _lingjing_usage_from_role(usage, resource_type, resource_index),
        "source": source,
    }
    reference_key = resource.get("reference_key") or resource.get("referenceKey")
    if reference_key:
        normalized["reference_key"] = reference_key
    return normalized

def _lingjing_resources_from_unified(resources: Any) -> list[dict[str, Any]]:
    if not isinstance(resources, list):
        return []

    normalized_resources: list[dict[str, Any]] = []
    for resource in resources:
        normalized = _lingjing_resource_from_unified(resource, len(normalized_resources))
        if normalized:
            normalized_resources.append(normalized)
    return normalized_resources

def _convert_content_generation_body_to_lingjing(
    request_body: dict[str, Any],
    *,
    model_code: str,
) -> dict[str, Any]:
    if "taskParams" in request_body or "modelCode" in request_body:
        payload = dict(request_body)
        payload["modelCode"] = model_code
        for key in ("model", "request_model", "provider", "provider_options", "route"):
            payload.pop(key, None)
        return payload

    prompt_parts: list[str] = []
    resources: list[dict[str, Any]] = []
    content = request_body.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip()
            if part_type == "text":
                text = str(part.get("text") or "").strip()
                if text:
                    prompt_parts.append(text)
                continue

            resource_type = ""
            url = ""
            if part_type == "image_url":
                resource_type = "image"
                url = _extract_url_from_content_part(part, "image_url")
            elif part_type == "video_url":
                resource_type = "video"
                url = _extract_url_from_content_part(part, "video_url")
            elif part_type == "audio_url":
                resource_type = "audio"
                url = _extract_url_from_content_part(part, "audio_url")

            if resource_type and url:
                resource: dict[str, Any] = {
                    "type": resource_type,
                    "usage": _lingjing_usage_from_role(part.get("role"), resource_type, len(resources)),
                    "source": _lingjing_source_from_value(url),
                }
                reference_key = part.get("reference_key")
                if reference_key:
                    resource["reference_key"] = reference_key
                resources.append(resource)

    input_payload: dict[str, Any] = {
        "prompt": str(request_body.get("prompt") or "\n".join(prompt_parts)).strip(),
    }

    quality = request_body.get("quality")
    if quality is None:
        resolution = str(request_body.get("resolution") or "").strip().lower()
        quality = resolution[:-1] if resolution.endswith("p") else resolution
    if quality:
        input_payload["quality"] = str(quality)

    for key in ("duration", "ratio", "resources", "generate_num", "prompt_optimizer"):
        if key in request_body and request_body.get(key) is not None:
            if key != "resources":
                input_payload[key] = request_body[key]
    unified_resources = _lingjing_resources_from_unified(request_body.get("resources"))
    if unified_resources:
        input_payload["resources"] = unified_resources
    elif resources:
        input_payload["resources"] = resources
    for key, value in _video_provider_options(request_body, "lingjing").items():
        if value is not None:
            input_payload[key] = value
    if "generate_audio" in request_body:
        input_payload["need_audio"] = bool(request_body.get("generate_audio"))
    if "need_audio" in request_body:
        input_payload["need_audio"] = bool(request_body.get("need_audio"))
    if "audio" in request_body:
        input_payload["need_audio"] = bool(request_body.get("audio"))

    return {"modelCode": model_code, "taskParams": {"input": input_payload}}

def _content_part_from_resource(resource: Any) -> Optional[dict[str, Any]]:
    if not isinstance(resource, dict):
        return None
    resource_type = str(resource.get("type") or "image").strip().lower()
    if resource_type not in {"image", "video", "audio"}:
        return None
    value = resource.get("url") or resource.get("value")
    source = resource.get("source")
    if not value and isinstance(source, dict):
        value = source.get("value")
    if not value:
        asset_id = resource.get("asset_id") or resource.get("assetId")
        if asset_id:
            value = f"asset://{asset_id}"
    if not value:
        return None

    key = f"{resource_type}_url"
    part: dict[str, Any] = {
        "type": key,
        key: {"url": str(value)},
    }
    role = resource.get("role") or resource.get("usage")
    if role:
        part["role"] = role
    return part

def _convert_video_body_to_content_generation(
    request_body: dict[str, Any],
    *,
    model_name: str,
    provider_name: str,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in request_body.items()
        if key not in {"provider", "provider_options", "route", "prompt", "resources", "audio"}
    }
    payload["model"] = model_name

    if not isinstance(payload.get("content"), list):
        content: list[dict[str, Any]] = []
        prompt = _video_prompt_from_body(request_body)
        if prompt:
            content.append({"type": "text", "text": prompt})
        for resource in request_body.get("resources") or []:
            part = _content_part_from_resource(resource)
            if part:
                content.append(part)
        if content:
            payload["content"] = content

    if "audio" in request_body and "generate_audio" not in payload:
        payload["generate_audio"] = bool(request_body.get("audio"))

    for key, value in _video_provider_options(request_body, provider_name).items():
        if value is not None:
            payload[key] = value

    return payload

def _lingjing_task_id_from_submit_response(obj: dict[str, Any]) -> Optional[str]:
    data = obj.get("data")
    if isinstance(data, dict):
        for key in ("taskId", "task_id"):
            value = data.get(key)
            if value:
                return str(value)
    return None

def _lingjing_status_to_content_status(status: Any) -> str:
    normalized = str(status or "").strip().upper()
    if normalized == "SUCCESS":
        return "succeeded"
    if normalized == "CANCELED":
        return "cancelled"
    if normalized in {"FAIL", "FAILED", "UNKNOWN"}:
        return "failed"
    if normalized in {"WAITING", "QUEUED", "SUBMITTED", "RUNNING"}:
        return "running"
    return normalized.lower() if normalized else "running"

def _first_lingjing_result_url(result: Any) -> Optional[str]:
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("url"):
                return str(item["url"])
    if isinstance(result, dict) and result.get("url"):
        return str(result["url"])
    return None

def _normalize_lingjing_content_generation_response(
    *,
    method: str,
    raw: bytes,
    task_id: Optional[str],
    request_model_name: str,
) -> tuple[bytes, Optional[str]]:
    obj = _maybe_json_object(raw)
    if not obj:
        return raw, None

    if method.upper() == "POST":
        upstream_task_id = _lingjing_task_id_from_submit_response(obj)
        if not upstream_task_id:
            return raw, None
        return _json_bytes(
            {
                "id": upstream_task_id,
                "model": request_model_name,
                "status": "queued",
                "created_at": int(time()),
                "upstream": obj,
            }
        ), upstream_task_id

    if method.upper() == "GET":
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        upstream_task_id = str(data.get("task_id") or task_id or "")
        result_url = _first_lingjing_result_url(data.get("result"))
        content: dict[str, Any] = {}
        if result_url:
            content["video_url"] = result_url
        normalized: dict[str, Any] = {
            "id": upstream_task_id,
            "model": request_model_name,
            "status": _lingjing_status_to_content_status(data.get("status")),
            "content": content,
            "upstream": obj,
        }
        if data.get("external_error"):
            normalized["error"] = {"message": data.get("external_error")}
        return _json_bytes(normalized), upstream_task_id or None

    return raw, None

def _usage_to_video_usage(usage: Any) -> Optional[dict[str, Any]]:
    if not isinstance(usage, dict):
        return None

    total_tokens = usage.get("total_tokens")
    completion_tokens = usage.get("completion_tokens")
    video_tokens = usage.get("video_tokens")
    if video_tokens is None:
        video_tokens = completion_tokens if completion_tokens is not None else total_tokens
    if total_tokens is None:
        total_tokens = video_tokens

    normalized: dict[str, Any] = {}
    if video_tokens is not None:
        normalized["video_tokens"] = video_tokens
        normalized["completion_tokens"] = video_tokens
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens
    for key, value in usage.items():
        normalized.setdefault(key, value)
    return normalized or None

def _positive_int_from_video_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(float(str(value).strip().rstrip("pP")))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None

def _video_resolution_height(request_body: dict[str, Any]) -> int:
    quality = request_body.get("quality")
    if quality is None:
        quality = request_body.get("resolution")
    return _positive_int_from_video_value(quality) or 720

def _estimated_video_usage_from_request(request_body: Optional[dict[str, Any]]) -> Optional[dict[str, int]]:
    if not isinstance(request_body, dict):
        return None

    duration = _positive_int_from_video_value(request_body.get("duration")) or 5
    fps = (
        _positive_int_from_video_value(request_body.get("fps"))
        or _positive_int_from_video_value(request_body.get("framespersecond"))
        or 24
    )
    resolution_height = _video_resolution_height(request_body)

    # Matches upstream token accounting for 720p Seedance 2.0 videos:
    # 5 seconds * 24 fps => 108900 video tokens.
    tokens_per_frame_720p = 907.5
    resolution_scale = (resolution_height / 720) ** 2
    video_tokens = max(1, int(round(duration * fps * tokens_per_frame_720p * resolution_scale)))
    return {
        "video_tokens": video_tokens,
        "completion_tokens": video_tokens,
        "total_tokens": video_tokens,
    }

def _normalize_video_task_response(
    *,
    method: str,
    raw: bytes,
    task_id: Optional[str],
    request_model_name: str,
    provider_name: str,
    is_lingjing: bool,
    estimated_usage: Optional[dict[str, Any]] = None,
) -> tuple[bytes, Optional[str]]:
    obj = _maybe_json_object(raw)
    if not obj:
        return raw, None

    method_upper = method.upper()
    if is_lingjing:
        if method_upper == "POST":
            upstream_task_id = _lingjing_task_id_from_submit_response(obj)
            if not upstream_task_id:
                return raw, None
            return _json_bytes(
                {
                    "id": upstream_task_id,
                    "model": request_model_name,
                    "provider": provider_name,
                    "status": "queued",
                    "created_at": int(time()),
                }
            ), upstream_task_id

        if method_upper == "GET":
            data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
            upstream_task_id = str(data.get("task_id") or data.get("taskId") or task_id or "")
            result_url = _first_lingjing_result_url(data.get("result"))
            normalized: dict[str, Any] = {
                "id": upstream_task_id,
                "model": request_model_name,
                "provider": provider_name,
                "status": _lingjing_status_to_content_status(data.get("status")),
                "video": {},
            }
            if result_url:
                normalized["video"]["url"] = result_url
            usage = _usage_to_video_usage(data.get("usage") if isinstance(data, dict) else None)
            if not usage and normalized["status"] == "succeeded":
                usage = _usage_to_video_usage(estimated_usage)
            if usage:
                normalized["usage"] = usage
            if data.get("external_error"):
                normalized["error"] = {"message": data.get("external_error")}
            return _json_bytes(normalized), upstream_task_id or None

        return raw, None

    if method_upper == "POST":
        upstream_task_id = obj.get("id")
        if not upstream_task_id:
            return raw, None
        return _json_bytes(
            {
                "id": str(upstream_task_id),
                "model": request_model_name,
                "provider": provider_name,
                "status": str(obj.get("status") or "queued"),
                "created_at": obj.get("created_at") or int(time()),
            }
        ), str(upstream_task_id)

    if method_upper == "GET":
        upstream_task_id = str(obj.get("id") or task_id or "")
        status = obj.get("status")
        if not upstream_task_id or not status:
            return raw, upstream_task_id or None

        video: dict[str, Any] = {}
        content = obj.get("content")
        if isinstance(content, dict) and content.get("video_url"):
            video["url"] = content.get("video_url")
        if obj.get("duration") is not None:
            video["duration"] = obj.get("duration")
        if obj.get("resolution") is not None:
            video["resolution"] = obj.get("resolution")
        if obj.get("ratio") is not None:
            video["ratio"] = obj.get("ratio")
        fps = obj.get("fps", obj.get("framespersecond"))
        if fps is not None:
            video["fps"] = fps

        normalized = {
            "id": upstream_task_id,
            "model": request_model_name,
            "provider": provider_name,
            "status": str(status),
            "video": video,
        }
        usage = _usage_to_video_usage(obj.get("usage"))
        if not usage and normalized["status"] == "succeeded":
            usage = _usage_to_video_usage(estimated_usage)
        if usage:
            normalized["usage"] = usage
        for key in ("created_at", "updated_at", "seed"):
            if obj.get(key) is not None:
                normalized[key] = obj[key]
        return _json_bytes(normalized), upstream_task_id

    return raw, None

def _lingjing_request_model_for_openapi(payload: Optional[dict[str, Any]], query_params: Any = None) -> str:
    if query_params is not None:
        raw_model = query_params.get("model")
        if raw_model:
            return str(raw_model).strip()

    body = payload or {}
    raw_model = body.get("model") or body.get("request_model")
    if raw_model:
        return str(raw_model).strip()

    model_code = str(body.get("modelCode") or "").strip()
    model_code_map = {
        "sd_2_0": "seedance-2-0",
        "sd_2_0_fast": "seedance-2-0-fast",
    }
    if model_code:
        request_model = model_code_map.get(model_code)
        if not request_model:
            raise HTTPException(status_code=400, detail=f"Unsupported Lingjing modelCode: {model_code}")
        return request_model

    return LINGJING_DEFAULT_REQUEST_MODEL

HOP_BY_HOP_RESPONSE_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

def _copy_upstream_response_headers(headers: Any) -> dict[str, str]:
    copied: dict[str, str] = {}
    if not headers:
        return copied
    for key, value in headers.items():
        normalized_key = str(key).lower()
        if normalized_key in HOP_BY_HOP_RESPONSE_HEADERS:
            continue
        copied[str(key)] = str(value)
    return copied

async def _prime_passthrough_upstream_stream(
    upstream_iter,
    *,
    disconnect_event: Optional[asyncio.Event] = None,
) -> list[bytes]:
    buffered_chunks: list[bytes] = []
    while True:
        if disconnect_event is not None and disconnect_event.is_set():
            return buffered_chunks

        try:
            chunk = await upstream_iter.__anext__()
        except StopAsyncIteration:
            if not buffered_chunks:
                raise HTTPException(status_code=502, detail="Upstream closed stream without data")
            return buffered_chunks

        buffered_chunks.append(chunk)
        if chunk:
            return buffered_chunks

def _log_model_names(request_model_name: Any, actual_model_name: Any = None) -> tuple[str, str]:
    request_model = str(request_model_name or "-")
    actual_model = str(actual_model_name or request_model)
    return request_model, actual_model

def _responses_request_id(current_info: Any) -> str:
    if isinstance(current_info, dict):
        request_id = current_info.get("request_id")
        if request_id:
            return str(request_id)
    return "-"

def _mask_secret_for_log(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "-"
    if len(raw) <= 10:
        return "***"
    return f"{raw[:4]}...{raw[-4:]}"

def _log_responses_downstream_disconnect(
    endpoint: str,
    current_info: Any,
    *,
    model_id: str,
    provider_name: Optional[str] = None,
    stage: str,
) -> None:
    trace_logger.info(
        "%s downstream disconnect stage=%s request_id=%s model=%s provider=%s",
        endpoint,
        stage,
        _responses_request_id(current_info),
        model_id,
        provider_name or "-",
    )

RESPONSES_STREAM_NETWORK_ERRORS = UPSTREAM_NETWORK_ERRORS

RESPONSES_FAILURE_STATUS_BY_CODE = {
    "account_deactivated": 403,
    "account_disabled": 403,
    "account_suspended": 403,
    "authentication_error": 401,
    "billing_hard_limit_reached": 429,
    "context_length_exceeded": 400,
    "deactivated_workspace": 403,
    "incorrect_api_key_provided": 401,
    "insufficient_quota": 429,
    "invalid_api_key": 401,
    "invalid_request_error": 400,
    "invalid_type": 400,
    "model_not_found": 404,
    "not_found_error": 404,
    "permission_denied": 403,
    "rate_limit_exceeded": 429,
    "unsupported_parameter": 400,
    "user_deactivated": 403,
    "user_suspended": 403,
}

RESPONSES_FAILURE_STATUS_BY_TYPE = {
    "authentication_error": 401,
    "invalid_request_error": 400,
    "not_found_error": 404,
    "permission_error": 403,
    "rate_limit_error": 429,
    "tokens": 429,
}

def _extract_responses_stream_event(raw_event: str) -> tuple[str, Any]:
    return parse_sse_event(raw_event)

RESPONSES_STREAM_PREFLIGHT_EVENTS = frozenset(
    {
        "response.created",
        "response.in_progress",
        "response.queued",
        "keepalive",
    }
)

def _encode_responses_sse_event(event_type: str, payload: Any) -> bytes:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
    ).encode("utf-8")

def _raw_responses_sse_event_bytes(raw_event: str) -> bytes:
    return raw_event.encode("utf-8") + b"\n\n"

def _build_responses_stream_keepalive_event() -> bytes:
    return _encode_responses_sse_event(
        "keepalive",
        {"type": "keepalive", "sequence_number": 0},
    )

def _build_responses_stream_error_event(status_code: int, error_message: Any) -> bytes:
    return _encode_responses_sse_event(
        "error",
        {
            "type": "error",
            "error": {
                "message": str(error_message),
                "status_code": int(status_code),
            },
        },
    )

def _stream_error_event_from_response(response: Any) -> bytes:
    status_code = int(getattr(response, "status_code", 500) or 500)
    body = getattr(response, "body", b"")
    if isinstance(body, bytes):
        message = body.decode("utf-8", errors="replace")
    else:
        message = str(body or f"Upstream request failed with status {status_code}")
    return _build_responses_stream_error_event(status_code, message)

def _responses_usage_from_payload(payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    usage = safe_get(payload, "response", "usage", default=None)
    if not isinstance(usage, dict):
        usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None

def _responses_part_has_text(part: Any) -> bool:
    if not isinstance(part, dict):
        return False

    text = part.get("text")
    if isinstance(text, str) and text:
        return True

    refusal = part.get("refusal")
    return isinstance(refusal, str) and bool(refusal)

def _responses_item_has_substantive_output(item: Any) -> bool:
    if not isinstance(item, dict):
        return False

    content = item.get("content")
    if isinstance(content, list) and any(_responses_part_has_text(part) for part in content):
        return True

    item_type = str(item.get("type") or "")
    if item_type in {"function_call", "tool_call"}:
        return bool(item.get("name") or item.get("arguments") or item.get("call_id"))

    return False

def _responses_stream_event_has_real_output(event_type: str, payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    if event_type.startswith("response.") and event_type.endswith(".delta"):
        return bool(str(payload.get("delta") or ""))

    if event_type in {"response.content_part.added", "response.content_part.done"}:
        return _responses_part_has_text(payload.get("part"))

    if event_type == "response.output_item.done":
        return _responses_item_has_substantive_output(payload.get("item"))

    if event_type.startswith("response.") and event_type.endswith(".done"):
        return bool(str(payload.get("text") or payload.get("refusal") or payload.get("arguments") or ""))

    return False

def _responses_stream_event_commits(event_type: str, payload: Any, commit_policy: str) -> bool:
    if event_type in RESPONSES_STREAM_PREFLIGHT_EVENTS:
        return False

    completed_with_usage = event_type == "response.completed" and _responses_usage_from_payload(payload) is not None
    if commit_policy == "completed_usage":
        return completed_with_usage
    return completed_with_usage or _responses_stream_event_has_real_output(event_type, payload)

def _responses_error_status_code(error_obj: Any) -> int:
    if isinstance(error_obj, dict):
        raw_status = error_obj.get("status_code") or error_obj.get("status")
        try:
            status_code = int(raw_status)
        except (TypeError, ValueError):
            status_code = None
        if status_code is not None and 100 <= status_code <= 599:
            return status_code

        error_code = str(error_obj.get("code") or "").strip().lower()
        if error_code in RESPONSES_FAILURE_STATUS_BY_CODE:
            return RESPONSES_FAILURE_STATUS_BY_CODE[error_code]

        error_type = str(error_obj.get("type") or "").strip().lower()
        if error_type in RESPONSES_FAILURE_STATUS_BY_TYPE:
            return RESPONSES_FAILURE_STATUS_BY_TYPE[error_type]

        message = str(error_obj.get("message") or "").lower()
        if "rate limit" in message or "too many requests" in message:
            return 429
        if "invalid" in message or "unsupported" in message:
            return 400
        if "not found" in message:
            return 404
        if "permission" in message or "forbidden" in message:
            return 403
        if "auth" in message or "api key" in message or "unauthorized" in message:
            return 401

    return 500

def _responses_failure_http_exception(payload: Any) -> Optional[HTTPException]:
    if not isinstance(payload, dict):
        return None

    error_obj = None
    response_status = str(safe_get(payload, "response", "status", default="") or "").strip().lower()
    payload_status = str(payload.get("status") or "").strip().lower()
    payload_type = str(payload.get("type") or "").strip().lower()

    if payload_type == "error" and payload.get("error") is not None:
        error_obj = payload.get("error")
    elif payload_type == "response.failed":
        error_obj = safe_get(payload, "response", "error", default=None)
    elif payload_status == "failed":
        error_obj = payload.get("error")
    elif response_status == "failed":
        error_obj = safe_get(payload, "response", "error", default=None)
    elif isinstance(payload.get("error"), dict):
        error_obj = payload.get("error")

    if error_obj is None and (payload_status == "failed" or response_status == "failed"):
        error_obj = {"message": "Responses upstream returned status=failed"}

    if error_obj is None:
        return None

    error_body = {"error": error_obj}
    return HTTPException(
        status_code=_responses_error_status_code(error_obj),
        detail=json.dumps(error_body, ensure_ascii=False),
    )

async def _prime_responses_upstream_stream(
    upstream_iter,
    *,
    disconnect_event: Optional[asyncio.Event] = None,
    commit_policy: str = "real_output",
    precommit_keepalive_callback: Optional[Callable[[Optional[bytes]], Awaitable[bool]]] = None,
) -> tuple[list[bytes], bool]:
    """
    Buffer structural Responses events until we see substantive output or a
    completed response with usage. Optional precommit keepalive emission does
    not commit the real Responses stream.
    """
    buffered_chunks: list[bytes] = []
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    sse_parser = IncrementalSSEParser()
    commit_policy = (commit_policy or "real_output").strip().lower()
    if commit_policy not in {"real_output", "completed_usage"}:
        commit_policy = "real_output"

    while True:
        if disconnect_event is not None and disconnect_event.is_set():
            return buffered_chunks, False

        try:
            chunk = await upstream_iter.__anext__()
        except StopAsyncIteration:
            if not buffered_chunks:
                raise HTTPException(status_code=502, detail="Upstream closed stream without data")
            if sse_parser.pending_text.strip():
                raise HTTPException(status_code=502, detail="Upstream closed stream with an incomplete SSE event")
            raise HTTPException(status_code=502, detail="Responses upstream closed before substantive output")

        decoded_chunk = decoder.decode(chunk)
        raw_events = sse_parser.feed(decoded_chunk)

        for event_index, raw_event in enumerate(raw_events):
            if not raw_event.strip():
                continue

            event_type, event_payload = _extract_responses_stream_event(raw_event)
            if event_type == "[DONE]":
                raise HTTPException(
                    status_code=502,
                    detail="Responses upstream ended before substantive output",
                )

            semantic_failure = _responses_failure_http_exception(event_payload)
            if semantic_failure is not None:
                raise semantic_failure

            event_bytes = _raw_responses_sse_event_bytes(raw_event)
            if event_type == "keepalive" and precommit_keepalive_callback is not None:
                handled = await precommit_keepalive_callback(event_bytes)
                if not handled:
                    buffered_chunks.append(event_bytes)
            else:
                if event_type == "response.created" and precommit_keepalive_callback is not None:
                    await precommit_keepalive_callback(None)
                buffered_chunks.append(event_bytes)

            if _responses_stream_event_commits(event_type, event_payload, commit_policy):
                for remaining_raw_event in raw_events[event_index + 1:]:
                    if remaining_raw_event.strip():
                        buffered_chunks.append(_raw_responses_sse_event_bytes(remaining_raw_event))
                if sse_parser.pending_text:
                    buffered_chunks.append(sse_parser.pending_text.encode("utf-8"))
                return buffered_chunks, True

            continue

class ResponsesRequestHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def request_responses(
        self,
        http_request: Request,
        request_data: ResponsesRequest,
        api_index: int,
        background_tasks: BackgroundTasks,
        endpoint: str = "/v1/responses",
    ):
        config = app.state.config
        request_model_name = request_data.model
        if not safe_get(config, 'api_keys', api_index, 'model'):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = request_info.get()
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_id = _responses_request_id(current_info)
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=endpoint,
            debug=is_debug,
            provider_resolver=get_right_order_providers,
        )
        _record_plan_observability(current_info, plan)
        runner = UpstreamRunner(
            plan,
            endpoint=endpoint,
            debug=is_debug,
            clear_provider_auth_cache=lambda provider_api_key_raw: _codex_oauth_cache.pop(provider_api_key_raw, None),
        )
        stream_output_queue: Optional[asyncio.Queue] = None
        stream_done_sentinel = object()
        stream_body_started = False
        stream_keepalive_sent = False
        stream_stats_tasks: list[asyncio.Task] = []

        async def emit_stream_chunk(chunk: Any) -> None:
            nonlocal stream_body_started
            if stream_output_queue is None:
                return
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            if not isinstance(chunk, (bytes, bytearray)):
                chunk = str(chunk).encode("utf-8")
            stream_body_started = True
            await stream_output_queue.put(bytes(chunk))

        async def emit_precommit_keepalive(upstream_keepalive: Optional[bytes]) -> bool:
            nonlocal stream_keepalive_sent
            if stream_output_queue is None:
                return False
            if stream_keepalive_sent:
                return True
            await emit_stream_chunk(upstream_keepalive or _build_responses_stream_keepalive_event())
            stream_keepalive_sent = True
            return True

        def schedule_channel_stats(channel_id: str, *, success: bool, provider_api_key: Optional[str]) -> None:
            args = (
                current_info["request_id"],
                channel_id,
                request_model_name,
                current_info["api_key"],
            )
            kwargs = {"success": success, "provider_api_key": provider_api_key}
            if stream_output_queue is not None:
                stream_stats_tasks.append(asyncio.create_task(update_channel_stats(*args, **kwargs)))
            else:
                background_tasks.add_task(update_channel_stats, *args, **kwargs)

        async def before_next_attempt():
            if disconnect_event is not None and disconnect_event.is_set():
                _log_responses_downstream_disconnect(
                    endpoint,
                    current_info,
                    model_id=request_model_name,
                    stage="before-provider-select",
                )
                return Response(content="", status_code=499)
            return None

        async def prepare_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            engine, stream_mode = get_engine(provider, endpoint=endpoint, original_model=original_model)
            if stream_mode is not None:
                request_data.stream = stream_mode

            attempt.state["failure_stage"] = "validation"
            if engine not in ("gpt", "codex"):
                raise HTTPException(
                    status_code=400,
                    detail=f"{endpoint} only supports upstream engines: gpt/codex (got {engine})",
                )

            wants_compact = endpoint.rstrip("/").endswith("/compact")
            if wants_compact:
                upstream_url = _normalize_responses_compact_upstream_url(provider.get("base_url", ""), engine)
            else:
                upstream_url = _normalize_responses_upstream_url(provider.get("base_url", ""), engine)

            if engine == "gpt" and "v1/responses" not in upstream_url:
                raise HTTPException(
                    status_code=400,
                    detail=f"{endpoint} requires provider base_url ending with /v1/responses (got {upstream_url})",
                )
            if wants_compact and "compact" not in upstream_url:
                raise HTTPException(
                    status_code=400,
                    detail=f"{endpoint} requires provider base_url ending with /v1/responses/compact (got {upstream_url})",
                )

            proxy = safe_get(config, "preferences", "proxy", default=None)
            proxy = safe_get(provider, "preferences", "proxy", default=proxy)
            channel_id = f"{provider_name}"
            trace = current_info.get("trace") if isinstance(current_info, dict) else None
            if isinstance(trace, RequestTrace):
                trace.mark("provider_selected")
                trace.set_tag("provider", channel_id)
                trace.set_tag("model", request_model_name)
            commit_policy = safe_get(
                provider,
                "preferences",
                "responses_stream_commit_policy",
                default="real_output",
            )
            attempt.state["upstream_url"] = upstream_url
            attempt.state["channel_id"] = channel_id
            attempt.state["engine"] = engine
            attempt.state["responses_stream_commit_policy"] = str(commit_policy or "real_output")
            attempt.state["failure_stage"] = "auth"

            attempt.provider_api_key_raw = await runner.select_provider_api_key(attempt)
            if isinstance(trace, RequestTrace):
                trace.mark("provider_key_selected")
            api_key = attempt.provider_api_key_raw
            codex_account_id = None
            if engine == "codex" and attempt.provider_api_key_raw:
                api_key, codex_account_id = await _resolve_codex_upstream_auth(
                    provider_name,
                    attempt.provider_api_key_raw,
                    proxy,
                )

            timeout_value = get_preference(
                app.state.provider_timeouts,
                provider_name,
                (original_model, request_model_name),
                DEFAULT_TIMEOUT,
            )
            attempt.state["proxy"] = proxy
            attempt.state["api_key"] = api_key
            attempt.state["codex_account_id"] = codex_account_id
            attempt.state["wants_compact"] = wants_compact
            attempt.state["timeout_value"] = int(timeout_value)

        async def execute_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            engine = attempt.state["engine"]
            proxy = attempt.state["proxy"]
            api_key = attempt.state["api_key"]
            codex_account_id = attempt.state["codex_account_id"]
            wants_compact = attempt.state["wants_compact"]
            timeout_value = attempt.state["timeout_value"]
            upstream_url = attempt.state["upstream_url"]
            channel_id = attempt.state["channel_id"]
            commit_policy = attempt.state.get("responses_stream_commit_policy", "real_output")

            headers = {
                "Content-Type": "application/json",
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            if engine == "codex":
                headers.setdefault("Openai-Beta", http_request.headers.get("Openai-Beta") or "responses=experimental")
                headers.setdefault("Originator", http_request.headers.get("Originator") or "codex_cli_rs")
                headers.setdefault("Version", CODEX_CLI_VERSION)
                headers.setdefault("Session_id", http_request.headers.get("Session_id") or str(uuid.uuid4()))
                headers.setdefault("User-Agent", CODEX_USER_AGENT)
                headers.setdefault("Accept", "text/event-stream" if request_data.stream else "application/json")
                if codex_account_id:
                    headers.setdefault("Chatgpt-Account-Id", str(codex_account_id))

            headers.update(safe_get(provider, "preferences", "headers", default={}) or {})
            if engine == "codex":
                force_codex_client_headers(headers)
            _add_trace_headers(headers, current_info)

            payload = request_data.model_dump(exclude_unset=True)
            payload["model"] = original_model
            if engine == "codex":
                payload.pop("previous_response_id", None)
                payload.pop("prompt_cache_retention", None)
                payload.pop("safety_identifier", None)
                payload.setdefault("instructions", "")

            apply_post_body_parameter_overrides(
                payload,
                provider,
                request_model_name,
                skip_keys={"translation_options"},
            )

            if engine == "codex":
                strip_unsupported_codex_payload_fields(payload, strip_store=wants_compact)

            _log_stdout_request_summary(channel_id, request_model_name, engine, plan.role)
            trace_logger.info(
                "endpoint=%s request_id=%s provider=%-11s model=%-22s engine=%-13s role=%s upstream_url=%s",
                endpoint,
                request_id,
                channel_id[:11],
                request_model_name,
                engine[:13],
                plan.role,
                upstream_url,
            )

            attempt.state["failure_stage"] = "upstream"
            attempt.state["track_channel_stats"] = True
            _log_debug_request_headers(
                "DEBUG upstream request headers",
                headers,
                endpoint=endpoint,
                upstream_url=upstream_url,
                provider=channel_id,
                model=request_model_name,
                actual_model=original_model,
            )
            _log_debug_request_body(
                "DEBUG upstream request body",
                payload,
                endpoint=endpoint,
                upstream_url=upstream_url,
                provider=channel_id,
                model=request_model_name,
                actual_model=original_model,
            )
            async with app.state.client_manager.get_client(upstream_url, proxy, http2=False if engine == "codex" else None) as client:
                json_payload = await asyncio.to_thread(json.dumps, payload)
                # json_payload = await asyncio.to_thread(json.dumps, payload, ensure_ascii=False)
                # if wants_compact:
                #     print("request /v1/responses/compact:", json_payload)
                if request_data.stream:
                    trace = current_info.get("trace") if isinstance(current_info, dict) else None
                    if isinstance(trace, RequestTrace):
                        trace.mark("upstream_send_start")
                    runtime_gauges.begin_waiting_first_byte(current_info)
                    stream_cm = client.stream("POST", upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
                    upstream_resp = await stream_cm.__aenter__()
                    if isinstance(trace, RequestTrace):
                        trace.mark("upstream_headers_received")
                    if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
                        runtime_gauges.end_waiting_first_byte(current_info)
                        raw = await upstream_resp.aread()
                        await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
                        try:
                            error_message = raw.decode("utf-8", errors="replace")
                        except Exception:
                            error_message = str(raw)
                        raise HTTPException(status_code=upstream_resp.status_code, detail=error_message)

                    upstream_iter = upstream_resp.aiter_bytes()
                    try:
                        buffered_chunks, stream_committed = await _prime_responses_upstream_stream(
                            upstream_iter,
                            disconnect_event=disconnect_event,
                            commit_policy=commit_policy,
                            precommit_keepalive_callback=emit_precommit_keepalive if stream_output_queue is not None else None,
                        )
                        _mark_first_byte_observed(current_info)
                    except HTTPException:
                        runtime_gauges.end_waiting_first_byte(current_info)
                        await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
                        raise
                    except RESPONSES_STREAM_NETWORK_ERRORS:
                        runtime_gauges.end_waiting_first_byte(current_info)
                        await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
                        raise
                    except BaseException:
                        runtime_gauges.end_waiting_first_byte(current_info)
                        await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
                        raise

                    if disconnect_event is not None and disconnect_event.is_set():
                        await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
                        _log_responses_downstream_disconnect(
                            endpoint,
                            current_info,
                            model_id=request_model_name,
                            provider_name=provider_name,
                            stage="before-stream-commit",
                        )
                        return Response(content="", status_code=499)

                    async def proxy_stream():
                        completed_seen = False
                        usage_seen = False
                        output_seen = False
                        proxy_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                        proxy_sse_parser = IncrementalSSEParser()

                        def track_responses_events(chunk: bytes) -> None:
                            nonlocal completed_seen, usage_seen, output_seen
                            decoded_chunk = proxy_decoder.decode(chunk)
                            for raw_event in proxy_sse_parser.feed(decoded_chunk):
                                if not raw_event.strip():
                                    continue

                                event_type, event_payload = _extract_responses_stream_event(raw_event)
                                if _responses_stream_event_has_real_output(event_type, event_payload):
                                    output_seen = True
                                if event_type == "response.completed":
                                    completed_seen = True
                                    if _responses_usage_from_payload(event_payload) is not None:
                                        usage_seen = True

                        try:
                            for chunk in buffered_chunks:
                                _mark_first_byte_observed(current_info)
                                if disconnect_event is not None and disconnect_event.is_set():
                                    _log_responses_downstream_disconnect(
                                        endpoint,
                                        current_info,
                                        model_id=request_model_name,
                                        provider_name=provider_name,
                                        stage="after-stream-commit",
                                    )
                                    return
                                track_responses_events(chunk)
                                yield chunk
                            async for chunk in upstream_iter:
                                _mark_first_byte_observed(current_info)
                                if disconnect_event is not None and disconnect_event.is_set():
                                    _log_responses_downstream_disconnect(
                                        endpoint,
                                        current_info,
                                        model_id=request_model_name,
                                        provider_name=provider_name,
                                        stage="after-stream-commit",
                                    )
                                    break
                                track_responses_events(chunk)
                                yield chunk
                        except RESPONSES_STREAM_NETWORK_ERRORS as e:
                            stream_stage = "post-commit" if stream_committed else "preflight"
                            error_text = str(e) or type(e).__name__
                            request_model, actual_model = _log_model_names(request_model_name, original_model)
                            trace_logger.warning(
                                "%s upstream stream aborted stage=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                                endpoint,
                                stream_stage,
                                type(e).__name__,
                                request_id,
                                request_model,
                                actual_model,
                                provider_name,
                                _mask_secret_for_log(attempt.provider_api_key_raw),
                                upstream_url,
                                error_text,
                            )
                            if stream_committed:
                                yield b"data: [DONE]\n\n"
                        finally:
                            if not completed_seen or not usage_seen:
                                trace_logger.warning(
                                    "%s upstream stream finished without completed usage request_id=%s model=%s provider=%s output_seen=%s completed_seen=%s usage_seen=%s upstream_url=%s",
                                    endpoint,
                                    request_id,
                                    request_model_name,
                                    provider_name,
                                    output_seen,
                                    completed_seen,
                                    usage_seen,
                                    upstream_url,
                                )
                            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)

                    schedule_channel_stats(
                        channel_id,
                        success=True,
                        provider_api_key=attempt.provider_api_key_raw,
                    )
                    current_info["first_response_time"] = 0
                    current_info["success"] = True
                    current_info["provider"] = channel_id
                    if stream_output_queue is not None:
                        async with aclosing(proxy_stream()) as upstream_body:
                            async for chunk in upstream_body:
                                await emit_stream_chunk(chunk)
                        return Response(status_code=204)
                    return StarletteStreamingResponse(proxy_stream(), media_type="text/event-stream")

                trace = current_info.get("trace") if isinstance(current_info, dict) else None
                if isinstance(trace, RequestTrace):
                    trace.mark("upstream_send_start")
                runtime_gauges.begin_waiting_first_byte(current_info)
                try:
                    upstream_resp = await client.post(upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
                    if isinstance(trace, RequestTrace):
                        trace.mark("upstream_headers_received")
                    _mark_first_byte_observed(current_info)
                except Exception:
                    runtime_gauges.end_waiting_first_byte(current_info)
                    raise
                if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
                    raw = await upstream_resp.aread()
                    try:
                        error_message = raw.decode("utf-8", errors="replace")
                    except Exception:
                        error_message = str(raw)
                    raise HTTPException(status_code=upstream_resp.status_code, detail=error_message)

                data = upstream_resp.json()
                semantic_failure = _responses_failure_http_exception(data)
                if semantic_failure is not None:
                    raise semantic_failure

                schedule_channel_stats(
                    channel_id,
                    success=True,
                    provider_api_key=attempt.provider_api_key_raw,
                )
                current_info["first_response_time"] = 0
                current_info["success"] = True
                current_info["provider"] = channel_id
                return JSONResponse(status_code=upstream_resp.status_code, content=data)

        def after_failure(attempt, exc, status_code, error_message):
            if attempt.state.get("track_channel_stats"):
                schedule_channel_stats(
                    attempt.state["channel_id"],
                    success=False,
                    provider_api_key=attempt.provider_api_key_raw,
                )

            upstream_url = attempt.state.get("upstream_url", "")
            failure_stage = attempt.state.get("failure_stage")
            request_model, actual_model = _log_model_names(request_model_name, attempt.original_model)
            if failure_stage == "auth" and isinstance(exc, ValueError):
                trace_logger.error(
                    "%s invalid codex api key request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                    endpoint,
                    request_id,
                    request_model,
                    actual_model,
                    attempt.provider_name,
                    _mask_secret_for_log(attempt.provider_api_key_raw),
                    upstream_url,
                    error_message,
                )
                return
            if failure_stage == "auth" and isinstance(exc, HTTPException):
                trace_logger.error(
                    "%s codex token refresh failed request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                    endpoint,
                    request_id,
                    request_model,
                    actual_model,
                    attempt.provider_name,
                    _mask_secret_for_log(attempt.provider_api_key_raw),
                    upstream_url,
                    error_message,
                )
                return

            trace_logger.error(
                "%s upstream error status=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                endpoint,
                status_code,
                type(exc).__name__,
                request_id,
                request_model,
                actual_model,
                attempt.state.get("channel_id", attempt.provider_name),
                _mask_secret_for_log(attempt.provider_api_key_raw),
                upstream_url,
                error_message,
            )

        def should_cool_down(exc, status_code, error_message, attempt):
            _ = error_message, attempt
            return not isinstance(exc, ValueError) and status_code not in (400, 413)

        def build_error_response(status_code, error_message):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            return build_upstream_error_response(
                status_code=status_code,
                error_message=error_message,
                fallback_prefix="Error: Current provider response failed",
            )

        def build_final_response(completed_plan):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            return JSONResponse(
                status_code=completed_plan.status_code,
                content={"error": f"All {request_model_name} error: {completed_plan.error_message}"},
            )

        async def run_responses_attempts():
            return await runner.run(
                execute_attempt,
                prepare_attempt=prepare_attempt,
                before_next_attempt=before_next_attempt,
                after_failure=after_failure,
                build_error_response=build_error_response,
                build_final_response=build_final_response,
                allow_channel_exclusion=True,
                should_cool_down=should_cool_down,
                on_retry=_record_retry_observability,
                on_cooldown=_record_cooldown_observability,
            )

        if request_data.stream:
            stream_output_queue = asyncio.Queue()

            async def stream_worker() -> None:
                try:
                    response = await run_responses_attempts()
                    if isinstance(response, Response):
                        if response.status_code == 204:
                            return
                        if hasattr(response, "body_iterator"):
                            async with aclosing(response.body_iterator):
                                async for chunk in response.body_iterator:
                                    await emit_stream_chunk(chunk)
                            return
                        if not stream_body_started:
                            await stream_output_queue.put(response)
                            return
                        if response.status_code != 499:
                            await emit_stream_chunk(_stream_error_event_from_response(response))
                            await emit_stream_chunk(b"data: [DONE]\n\n")
                    elif response is not None:
                        await emit_stream_chunk(response)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    trace_logger.error(
                        "%s stream worker failed request_id=%s model=%s error_type=%s: %s",
                        endpoint,
                        request_id,
                        request_model_name,
                        type(exc).__name__,
                        str(exc) or type(exc).__name__,
                    )
                    if not stream_body_started:
                        await stream_output_queue.put(
                            JSONResponse(
                                status_code=500,
                                content={"error": str(exc) or type(exc).__name__},
                            )
                        )
                    else:
                        await emit_stream_chunk(_build_responses_stream_error_event(500, str(exc) or type(exc).__name__))
                        await emit_stream_chunk(b"data: [DONE]\n\n")
                finally:
                    if stream_stats_tasks:
                        await asyncio.gather(*stream_stats_tasks, return_exceptions=True)
                    await stream_output_queue.put(stream_done_sentinel)

            worker_task = asyncio.create_task(stream_worker())
            first_item = await stream_output_queue.get()
            if first_item is stream_done_sentinel:
                return Response(content="", status_code=204)
            if isinstance(first_item, Response):
                with suppress(asyncio.CancelledError):
                    await worker_task
                return first_item

            async def stream_body():
                try:
                    yield first_item
                    while True:
                        item = await stream_output_queue.get()
                        if item is stream_done_sentinel:
                            break
                        yield item
                finally:
                    if disconnect_event is not None:
                        disconnect_event.set()
                    if not worker_task.done():
                        worker_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await worker_task

            return StarletteStreamingResponse(stream_body(), media_type="text/event-stream")

        return await run_responses_attempts()

class MessagesPassthroughHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def request_messages(
        self,
        http_request: Request,
        request_body: dict[str, Any],
        api_index: int,
        background_tasks: BackgroundTasks,
        endpoint: str = "/v1/messages",
    ):
        if not isinstance(request_body, dict):
            raise HTTPException(status_code=422, detail="Request body must be a JSON object")

        request_model_name = str(request_body.get("model") or "").strip()
        if not request_model_name:
            raise HTTPException(status_code=422, detail="Request body requires a model")

        config = app.state.config
        if not safe_get(config, "api_keys", api_index, "model"):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = request_info.get()
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_id = _responses_request_id(current_info)
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=endpoint,
            debug=is_debug,
            provider_resolver=get_right_order_providers,
        )
        _record_plan_observability(current_info, plan)
        runner = UpstreamRunner(
            plan,
            endpoint=endpoint,
            debug=is_debug,
        )
        last_error_response: dict[str, Any] = {}

        async def before_next_attempt():
            if disconnect_event is not None and disconnect_event.is_set():
                trace_logger.info(
                    "%s downstream disconnect stage=before-provider-select request_id=%s model=%s",
                    endpoint,
                    request_id,
                    request_model_name,
                )
                return Response(content="", status_code=499)
            return None

        async def prepare_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            engine, stream_mode = get_engine(provider, endpoint=endpoint, original_model=original_model)
            attempt.state["failure_stage"] = "validation"

            upstream_url = _normalize_messages_upstream_url(provider.get("base_url", ""))
            if not upstream_url:
                raise HTTPException(status_code=400, detail=f"{endpoint} requires provider base_url")

            upstream_path = urlparse(upstream_url).path.rstrip("/")
            is_messages_upstream = upstream_path.endswith("/v1/messages") or upstream_path.endswith("/messages")
            if engine != "claude" and not is_messages_upstream:
                raise HTTPException(
                    status_code=400,
                    detail=f"{endpoint} only supports upstream engine: claude (got {engine})",
                )
            engine = "claude"

            proxy = safe_get(config, "preferences", "proxy", default=None)
            proxy = safe_get(provider, "preferences", "proxy", default=proxy)
            channel_id = f"{provider_name}"

            attempt.state["upstream_url"] = upstream_url
            attempt.state["channel_id"] = channel_id
            attempt.state["engine"] = engine
            attempt.state["proxy"] = proxy
            attempt.state["stream_mode"] = stream_mode
            attempt.state["failure_stage"] = "auth"

            attempt.provider_api_key_raw = await runner.select_provider_api_key(attempt)

            timeout_value = get_preference(
                app.state.provider_timeouts,
                provider_name,
                (original_model, request_model_name),
                DEFAULT_TIMEOUT,
            )
            attempt.state["api_key"] = attempt.provider_api_key_raw
            attempt.state["timeout_value"] = int(timeout_value)

        async def execute_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            proxy = attempt.state["proxy"]
            api_key = attempt.state["api_key"]
            timeout_value = attempt.state["timeout_value"]
            upstream_url = attempt.state["upstream_url"]
            channel_id = attempt.state["channel_id"]

            payload = dict(request_body)
            payload["model"] = original_model
            if attempt.state.get("stream_mode") is not None:
                payload["stream"] = bool(attempt.state["stream_mode"])

            apply_post_body_parameter_overrides(payload, provider, request_model_name)

            headers = {
                "Content-Type": "application/json",
                "anthropic-version": http_request.headers.get("anthropic-version") or "2023-06-01",
            }
            anthropic_beta = http_request.headers.get("anthropic-beta")
            if anthropic_beta:
                headers["anthropic-beta"] = anthropic_beta
            if api_key:
                headers["x-api-key"] = str(api_key)
            headers.update(safe_get(provider, "preferences", "headers", default={}) or {})

            _log_stdout_request_summary(channel_id, request_model_name, "claude", plan.role)
            trace_logger.info(
                "endpoint=%s request_id=%s provider=%-11s model=%-22s engine=%-13s role=%s upstream_url=%s",
                endpoint,
                request_id,
                channel_id[:11],
                request_model_name,
                "claude",
                plan.role,
                upstream_url,
            )

            attempt.state["failure_stage"] = "upstream"
            attempt.state["track_channel_stats"] = True
            _log_debug_request_headers(
                "DEBUG upstream request headers",
                headers,
                endpoint=endpoint,
                upstream_url=upstream_url,
                provider=channel_id,
                model=request_model_name,
                actual_model=original_model,
            )
            _log_debug_request_body(
                "DEBUG upstream request body",
                payload,
                endpoint=endpoint,
                upstream_url=upstream_url,
                provider=channel_id,
                model=request_model_name,
                actual_model=original_model,
            )
            json_payload = await asyncio.to_thread(json.dumps, payload)
            wants_stream = bool(payload.get("stream"))

            async with app.state.client_manager.get_client(upstream_url, proxy) as client:
                if wants_stream:
                    stream_cm = client.stream("POST", upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
                    upstream_resp = await stream_cm.__aenter__()
                    response_headers = _copy_upstream_response_headers(upstream_resp.headers)
                    if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
                        raw = await upstream_resp.aread()
                        await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
                        last_error_response.clear()
                        last_error_response.update(
                            {
                                "body": raw,
                                "headers": response_headers,
                            }
                        )
                        raise HTTPException(
                            status_code=upstream_resp.status_code,
                            detail=raw.decode("utf-8", errors="replace"),
                        )

                    upstream_iter = upstream_resp.aiter_raw()
                    try:
                        buffered_chunks = await _prime_passthrough_upstream_stream(
                            upstream_iter,
                            disconnect_event=disconnect_event,
                        )
                    except BaseException:
                        await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
                        raise

                    if disconnect_event is not None and disconnect_event.is_set():
                        await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
                        trace_logger.info(
                            "%s downstream disconnect stage=before-stream-commit request_id=%s model=%s provider=%s",
                            endpoint,
                            request_id,
                            request_model_name,
                            provider_name,
                        )
                        return Response(content="", status_code=499)

                    async def proxy_stream():
                        try:
                            for chunk in buffered_chunks:
                                if disconnect_event is not None and disconnect_event.is_set():
                                    trace_logger.info(
                                        "%s downstream disconnect stage=after-stream-commit request_id=%s model=%s provider=%s",
                                        endpoint,
                                        request_id,
                                        request_model_name,
                                        provider_name,
                                    )
                                    return
                                yield chunk
                            async for chunk in upstream_iter:
                                if disconnect_event is not None and disconnect_event.is_set():
                                    trace_logger.info(
                                        "%s downstream disconnect stage=after-stream-commit request_id=%s model=%s provider=%s",
                                        endpoint,
                                        request_id,
                                        request_model_name,
                                        provider_name,
                                    )
                                    break
                                yield chunk
                        finally:
                            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)

                    background_tasks.add_task(
                        update_channel_stats,
                        current_info["request_id"],
                        channel_id,
                        request_model_name,
                        current_info["api_key"],
                        success=True,
                        provider_api_key=attempt.provider_api_key_raw,
                    )
                    current_info["first_response_time"] = 0
                    current_info["success"] = True
                    current_info["provider"] = channel_id
                    return StarletteStreamingResponse(
                        proxy_stream(),
                        status_code=upstream_resp.status_code,
                        headers=response_headers,
                        media_type=response_headers.get("content-type", "text/event-stream"),
                    )

                upstream_resp = await client.post(upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
                response_headers = _copy_upstream_response_headers(upstream_resp.headers)
                raw = upstream_resp.content
                if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
                    last_error_response.clear()
                    last_error_response.update(
                        {
                            "body": raw,
                            "headers": response_headers,
                        }
                    )
                    raise HTTPException(
                        status_code=upstream_resp.status_code,
                        detail=raw.decode("utf-8", errors="replace"),
                    )

                background_tasks.add_task(
                    update_channel_stats,
                    current_info["request_id"],
                    channel_id,
                    request_model_name,
                    current_info["api_key"],
                    success=True,
                    provider_api_key=attempt.provider_api_key_raw,
                )
                current_info["first_response_time"] = 0
                current_info["success"] = True
                current_info["provider"] = channel_id
                return Response(
                    content=raw,
                    status_code=upstream_resp.status_code,
                    headers=response_headers,
                    media_type=response_headers.get("content-type", "application/json"),
                )

        def after_failure(attempt, exc, status_code, error_message):
            if attempt.state.get("track_channel_stats"):
                background_tasks.add_task(
                    update_channel_stats,
                    current_info["request_id"],
                    attempt.state["channel_id"],
                    request_model_name,
                    current_info["api_key"],
                    success=False,
                    provider_api_key=attempt.provider_api_key_raw,
                )

            request_model, actual_model = _log_model_names(request_model_name, attempt.original_model)
            trace_logger.error(
                "%s upstream error status=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                endpoint,
                status_code,
                type(exc).__name__,
                request_id,
                request_model,
                actual_model,
                attempt.state.get("channel_id", attempt.provider_name),
                _mask_secret_for_log(attempt.provider_api_key_raw),
                attempt.state.get("upstream_url", ""),
                error_message,
            )

        def should_cool_down(exc, status_code, error_message, attempt):
            _ = exc, error_message, attempt
            return status_code not in (400, 413)

        def build_error_response(status_code, error_message):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            if last_error_response.get("body") is not None:
                headers = last_error_response.get("headers") or {}
                return Response(
                    content=last_error_response["body"],
                    status_code=status_code,
                    headers=headers,
                    media_type=headers.get("content-type", "application/json"),
                )
            return build_upstream_error_response(
                status_code=status_code,
                error_message=error_message,
                fallback_prefix="Error: Current provider response failed",
            )

        def build_final_response(completed_plan):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            return JSONResponse(
                status_code=completed_plan.status_code,
                content={"error": f"All {request_model_name} error: {completed_plan.error_message}"},
            )

        return await runner.run(
            execute_attempt,
            prepare_attempt=prepare_attempt,
            before_next_attempt=before_next_attempt,
            after_failure=after_failure,
            build_error_response=build_error_response,
            build_final_response=build_final_response,
            should_cool_down=should_cool_down,
            on_retry=_record_retry_observability,
            on_cooldown=_record_cooldown_observability,
        )

class VideoTaskHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)
        self.task_routes: dict[str, dict[str, Any]] = {}
        self.task_route_ttl_seconds = 7 * 24 * 60 * 60
        self.max_task_routes = 10000

    def _prune_task_routes(self) -> None:
        if not self.task_routes:
            return

        now = time()
        expired_ids = [
            task_id
            for task_id, route in self.task_routes.items()
            if now - float(route.get("created_at", now)) > self.task_route_ttl_seconds
        ]
        for task_id in expired_ids:
            self.task_routes.pop(task_id, None)

        overflow = len(self.task_routes) - self.max_task_routes
        if overflow <= 0:
            return

        oldest_task_ids = sorted(
            self.task_routes,
            key=lambda task_id: float(self.task_routes[task_id].get("created_at", now)),
        )[:overflow]
        for task_id in oldest_task_ids:
            self.task_routes.pop(task_id, None)

    def _remember_task_route(
        self,
        *,
        task_id: str,
        request_model_name: str,
        original_model: str,
        provider: dict,
        provider_name: str,
        provider_api_key_raw: Optional[str],
        client_api_key: Optional[str],
        estimated_usage: Optional[dict[str, Any]] = None,
    ) -> None:
        if not task_id:
            return
        self._prune_task_routes()
        self.task_routes[task_id] = {
            "created_at": time(),
            "request_model_name": request_model_name,
            "original_model": original_model,
            "provider": provider,
            "provider_name": provider_name,
            "provider_api_key_raw": provider_api_key_raw,
            "client_api_key": client_api_key,
            "estimated_usage": estimated_usage,
        }

    def _resolve_task_route(self, task_id: str, client_api_key: Optional[str]) -> Optional[dict[str, Any]]:
        route = self.task_routes.get(task_id)
        if route is None:
            return None
        if route.get("client_api_key") and client_api_key and route.get("client_api_key") != client_api_key:
            raise HTTPException(status_code=403, detail="Task belongs to a different API key")
        return route

    def _provider_resolver(self, request_body: Optional[dict[str, Any]]):
        requested_provider = _video_requested_provider(request_body)
        if not requested_provider:
            return get_right_order_providers

        async def resolve_video_providers(
            request_model_name: str,
            config: dict,
            api_index: int,
            scheduling_algorithm: str,
            api_list: list[str],
            models_list: dict[str, list[str]],
            **kwargs,
        ):
            providers = await get_right_order_providers(
                request_model_name,
                config,
                api_index,
                scheduling_algorithm,
                api_list,
                models_list,
                **kwargs,
            )
            filtered = [
                provider
                for provider in providers
                if str(provider.get("provider") or "").strip().lower() == requested_provider.lower()
            ]
            if not filtered:
                raise HTTPException(status_code=404, detail=f"No available provider for video task: {requested_provider}")
            return filtered

        return resolve_video_providers

    def _raw_response(
        self,
        upstream_resp: httpx.Response,
        raw: bytes,
        media_type: Optional[str] = None,
    ) -> Response:
        response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        if media_type:
            response_headers["content-type"] = media_type
        return Response(
            content=raw,
            status_code=upstream_resp.status_code,
            headers=response_headers,
            media_type=media_type or response_headers.get("content-type", "application/json"),
        )

    def _is_non_retryable_client_error(self, status_code: int) -> bool:
        return 400 <= status_code < 500 and status_code not in (401, 403, 408, 409, 425, 429)

    async def _send_upstream(
        self,
        *,
        method: str,
        upstream_url: str,
        headers: dict[str, str],
        payload: Optional[dict[str, Any]],
        proxy: Optional[str],
        timeout_value: int,
    ) -> httpx.Response:
        async with app.state.client_manager.get_client(upstream_url, proxy) as client:
            if method == "POST":
                json_payload = await asyncio.to_thread(json.dumps, payload or {})
                return await client.post(upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
            if method == "GET":
                return await client.get(upstream_url, headers=headers, timeout=timeout_value)
            if method == "PUT":
                json_payload = await asyncio.to_thread(json.dumps, payload or {})
                return await client.put(upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
            if method == "DELETE":
                return await client.delete(upstream_url, headers=headers, timeout=timeout_value)
        raise HTTPException(status_code=405, detail=f"Unsupported method: {method}")

    def _mark_result(
        self,
        *,
        background_tasks: BackgroundTasks,
        current_info: dict,
        channel_id: str,
        request_model_name: str,
        success: bool,
        provider_api_key_raw: Optional[str],
    ) -> None:
        if current_info is None:
            return
        current_info["first_response_time"] = 0 if success else -1
        current_info["success"] = success
        current_info["provider"] = channel_id if success else None
        current_info["model"] = current_info.get("model") or request_model_name
        background_tasks.add_task(
            update_channel_stats,
            current_info["request_id"],
            channel_id,
            request_model_name,
            current_info["api_key"],
            success=success,
            provider_api_key=provider_api_key_raw,
        )

    async def _request_with_fixed_route(
        self,
        *,
        method: str,
        task_id: str,
        route: dict[str, Any],
        background_tasks: BackgroundTasks,
        current_info: dict,
    ) -> Response:
        provider = route["provider"]
        provider_name = route["provider_name"]
        request_model_name = route["request_model_name"]
        original_model = route["original_model"]
        provider_api_key_raw = route.get("provider_api_key_raw")
        proxy = safe_get(app.state.config, "preferences", "proxy", default=None)
        proxy = safe_get(provider, "preferences", "proxy", default=proxy)
        timeout_value = get_preference(
            app.state.provider_timeouts,
            provider_name,
            (original_model, request_model_name),
            DEFAULT_TIMEOUT,
        )
        adapter = get_video_adapter(app.state.config, provider, provider_name)
        try:
            upstream_request = adapter.build_request(
                method=method,
                task_id=task_id,
                request_body=None,
                request_model_name=request_model_name,
                original_model=original_model,
                provider=provider,
                provider_name=provider_name,
                provider_api_key_raw=provider_api_key_raw,
            )
        except VideoAdapterError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        upstream_url = upstream_request.url
        headers = upstream_request.headers
        channel_id = f"{provider_name}"

        trace_logger.info(
            "endpoint=%s method=%s request_id=%s provider=%-11s model=%-22s engine=%-13s upstream_url=%s",
            CONTENT_GENERATION_TASKS_ENDPOINT,
            method,
            _responses_request_id(current_info),
            channel_id[:11],
            request_model_name,
            "content-generation",
            upstream_url,
        )

        try:
            upstream_resp = await self._send_upstream(
                method=upstream_request.method,
                upstream_url=upstream_url,
                headers=headers,
                payload=upstream_request.payload,
                proxy=proxy,
                timeout_value=int(timeout_value),
            )
            raw = upstream_resp.content
            normalized = adapter.normalize_response(
                method=method,
                raw=raw,
                task_id=task_id,
                request_model_name=request_model_name,
                provider_name=provider_name,
                estimated_usage=route.get("estimated_usage"),
            )
            raw = normalized.raw
            response_media_type = normalized.media_type if _maybe_json_object(raw) else None
            success = 200 <= upstream_resp.status_code < 300
            self._mark_result(
                background_tasks=background_tasks,
                current_info=current_info,
                channel_id=channel_id,
                request_model_name=request_model_name,
                success=success,
                provider_api_key_raw=provider_api_key_raw,
            )
            if success and method == "DELETE":
                self.task_routes.pop(task_id, None)
            return self._raw_response(upstream_resp, raw, media_type=response_media_type)
        except Exception:
            self._mark_result(
                background_tasks=background_tasks,
                current_info=current_info,
                channel_id=channel_id,
                request_model_name=request_model_name,
                success=False,
                provider_api_key_raw=provider_api_key_raw,
            )
            raise

    async def create_task(
        self,
        http_request: Request,
        request_body: dict[str, Any],
        api_index: int,
        background_tasks: BackgroundTasks,
    ):
        if not isinstance(request_body, dict):
            raise HTTPException(status_code=422, detail="Request body must be a JSON object")
        request_model_name = str(request_body.get("model") or "").strip()
        if not request_model_name:
            raise HTTPException(status_code=422, detail="Request body requires a model")
        return await self._request_with_model_route(
            http_request=http_request,
            request_model_name=request_model_name,
            request_body=request_body,
            api_index=api_index,
            background_tasks=background_tasks,
            method="POST",
            task_id=None,
        )

    async def get_or_delete_task(
        self,
        http_request: Request,
        task_id: str,
        api_index: int,
        background_tasks: BackgroundTasks,
        *,
        method: str,
        model: Optional[str] = None,
    ):
        current_info = request_info.get()
        route = self._resolve_task_route(task_id, current_info.get("api_key"))
        if route is not None:
            return await self._request_with_fixed_route(
                method=method,
                task_id=task_id,
                route=route,
                background_tasks=background_tasks,
                current_info=current_info,
            )

        request_model_name = str(model or "").strip()
        if not request_model_name:
            raise HTTPException(
                status_code=404,
                detail="Unknown content generation task id. Query with ?model=<model> if the task was created before this gateway instance learned the route.",
            )

        return await self._request_with_model_route(
            http_request=http_request,
            request_model_name=request_model_name,
            request_body=None,
            api_index=api_index,
            background_tasks=background_tasks,
            method=method,
            task_id=task_id,
        )

    async def _request_with_model_route(
        self,
        *,
        http_request: Request,
        request_model_name: str,
        request_body: Optional[dict[str, Any]],
        api_index: int,
        background_tasks: BackgroundTasks,
        method: str,
        task_id: Optional[str],
    ):
        _ = http_request
        config = app.state.config
        if not safe_get(config, "api_keys", api_index, "model"):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = request_info.get()
        current_info["model"] = current_info.get("model") or request_model_name
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_id = _responses_request_id(current_info)
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
            debug=is_debug,
            provider_resolver=self._provider_resolver(request_body),
        )
        _record_plan_observability(current_info, plan)
        runner = UpstreamRunner(
            plan,
            endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
            debug=is_debug,
        )
        last_error_response: dict[str, Any] = {}

        async def before_next_attempt():
            if disconnect_event is not None and disconnect_event.is_set():
                trace_logger.info(
                    "%s downstream disconnect stage=before-provider-select request_id=%s model=%s",
                    CONTENT_GENERATION_TASKS_ENDPOINT,
                    request_id,
                    request_model_name,
                )
                return Response(content="", status_code=499)
            return None

        async def prepare_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            engine, _ = get_engine(provider, endpoint=CONTENT_GENERATION_TASKS_ENDPOINT, original_model=original_model)
            attempt.state["failure_stage"] = "validation"
            if engine != "content-generation":
                raise HTTPException(
                    status_code=400,
                    detail=f"{CONTENT_GENERATION_TASKS_ENDPOINT} only supports upstream engine: content-generation (got {engine})",
                )

            proxy = safe_get(config, "preferences", "proxy", default=None)
            proxy = safe_get(provider, "preferences", "proxy", default=proxy)
            channel_id = f"{provider_name}"
            attempt.state["failure_stage"] = "auth"
            attempt.provider_api_key_raw = await runner.select_provider_api_key(attempt)

            adapter = get_video_adapter(config, provider, provider_name)
            try:
                upstream_request = adapter.build_request(
                    method=method,
                    task_id=task_id,
                    request_body=request_body,
                    request_model_name=request_model_name,
                    original_model=original_model,
                    provider=provider,
                    provider_name=provider_name,
                    provider_api_key_raw=attempt.provider_api_key_raw,
                )
            except VideoAdapterError as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
            if not upstream_request.url:
                raise HTTPException(status_code=400, detail=f"{CONTENT_GENERATION_TASKS_ENDPOINT} requires provider base_url")
            if upstream_request.payload is not None:
                apply_post_body_parameter_overrides(upstream_request.payload, provider, request_model_name)

            attempt.state["video_adapter"] = adapter
            attempt.state["upstream_request"] = upstream_request
            attempt.state["upstream_url"] = upstream_request.url
            attempt.state["channel_id"] = channel_id
            attempt.state["engine"] = engine
            attempt.state["proxy"] = proxy

            timeout_value = get_preference(
                app.state.provider_timeouts,
                provider_name,
                (original_model, request_model_name),
                DEFAULT_TIMEOUT,
            )
            attempt.state["api_key"] = attempt.provider_api_key_raw
            attempt.state["timeout_value"] = int(timeout_value)

        async def execute_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            proxy = attempt.state["proxy"]
            timeout_value = attempt.state["timeout_value"]
            adapter = attempt.state["video_adapter"]
            upstream_request = attempt.state["upstream_request"]
            upstream_url = attempt.state["upstream_url"]
            channel_id = attempt.state["channel_id"]
            payload = upstream_request.payload
            headers = upstream_request.headers

            _log_stdout_request_summary(channel_id, request_model_name, "content-generation", plan.role)
            trace_logger.info(
                "endpoint=%s method=%s request_id=%s provider=%-11s model=%-22s engine=%-13s role=%s upstream_url=%s",
                CONTENT_GENERATION_TASKS_ENDPOINT,
                method,
                request_id,
                channel_id[:11],
                request_model_name,
                "content-generation",
                plan.role,
                upstream_url,
            )
            attempt.state["failure_stage"] = "upstream"
            attempt.state["track_channel_stats"] = True

            _log_debug_request_headers(
                "DEBUG upstream request headers",
                headers,
                endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
                upstream_url=upstream_url,
                provider=channel_id,
                model=request_model_name,
                actual_model=original_model,
            )
            if payload is not None:
                _log_debug_request_body(
                    "DEBUG upstream request body",
                    payload,
                    endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
                    upstream_url=upstream_url,
                    provider=channel_id,
                    model=request_model_name,
                    actual_model=original_model,
                )

            upstream_resp = await self._send_upstream(
                method=upstream_request.method,
                upstream_url=upstream_url,
                headers=headers,
                payload=payload,
                proxy=proxy,
                timeout_value=timeout_value,
            )
            raw = upstream_resp.content
            normalized = adapter.normalize_response(
                method=method,
                raw=raw,
                task_id=task_id,
                request_model_name=request_model_name,
                provider_name=provider_name,
                estimated_usage=_estimated_video_usage_from_request(request_body),
            )
            raw = normalized.raw
            normalized_task_id = normalized.task_id
            response_media_type = normalized.media_type if _maybe_json_object(raw) else None
            if method == "POST" and normalized_task_id:
                self._remember_task_route(
                    task_id=normalized_task_id,
                    request_model_name=request_model_name,
                    original_model=original_model,
                    provider=provider,
                    provider_name=provider_name,
                    provider_api_key_raw=attempt.provider_api_key_raw,
                    client_api_key=current_info.get("api_key"),
                    estimated_usage=_estimated_video_usage_from_request(request_body),
                )
            if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
                if self._is_non_retryable_client_error(upstream_resp.status_code):
                    self._mark_result(
                        background_tasks=background_tasks,
                        current_info=current_info,
                        channel_id=channel_id,
                        request_model_name=request_model_name,
                        success=False,
                        provider_api_key_raw=attempt.provider_api_key_raw,
                    )
                    return self._raw_response(upstream_resp, raw, media_type=response_media_type)

                last_error_response.clear()
                last_error_response.update(
                    {
                        "body": raw,
                        "headers": _copy_upstream_response_headers(upstream_resp.headers),
                    }
                )
                raise HTTPException(
                    status_code=upstream_resp.status_code,
                    detail=raw.decode("utf-8", errors="replace"),
                )

            if method == "DELETE" and task_id:
                self.task_routes.pop(task_id, None)

            self._mark_result(
                background_tasks=background_tasks,
                current_info=current_info,
                channel_id=channel_id,
                request_model_name=request_model_name,
                success=True,
                provider_api_key_raw=attempt.provider_api_key_raw,
            )
            return self._raw_response(upstream_resp, raw, media_type=response_media_type)

        def after_failure(attempt, exc, status_code, error_message):
            if attempt.state.get("track_channel_stats"):
                background_tasks.add_task(
                    update_channel_stats,
                    current_info["request_id"],
                    attempt.state["channel_id"],
                    request_model_name,
                    current_info["api_key"],
                    success=False,
                    provider_api_key=attempt.provider_api_key_raw,
                )

            request_model, actual_model = _log_model_names(request_model_name, attempt.original_model)
            trace_logger.error(
                "%s upstream error status=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                CONTENT_GENERATION_TASKS_ENDPOINT,
                status_code,
                type(exc).__name__,
                request_id,
                request_model,
                actual_model,
                attempt.state.get("channel_id", attempt.provider_name),
                _mask_secret_for_log(attempt.provider_api_key_raw),
                attempt.state.get("upstream_url", ""),
                error_message,
            )

        def should_cool_down(exc, status_code, error_message, attempt):
            _ = exc, error_message, attempt
            return status_code in (401, 403, 429) or status_code >= 500

        def build_error_response(status_code, error_message):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            if last_error_response.get("body") is not None:
                headers = last_error_response.get("headers") or {}
                return Response(
                    content=last_error_response["body"],
                    status_code=status_code,
                    headers=headers,
                    media_type=headers.get("content-type", "application/json"),
                )
            return build_upstream_error_response(
                status_code=status_code,
                error_message=error_message,
                fallback_prefix="Error: Current provider response failed",
            )

        def build_final_response(completed_plan):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            return JSONResponse(
                status_code=completed_plan.status_code,
                content={"error": f"All {request_model_name} error: {completed_plan.error_message}"},
            )

        return await runner.run(
            execute_attempt,
            prepare_attempt=prepare_attempt,
            before_next_attempt=before_next_attempt,
            after_failure=after_failure,
            build_error_response=build_error_response,
            build_final_response=build_final_response,
            should_cool_down=should_cool_down,
            on_retry=_record_retry_observability,
            on_cooldown=_record_cooldown_observability,
        )

class LingjingOpenapiHandler:
    def __init__(self):
        self.last_provider_indices = defaultdict(lambda: -1)
        self.locks = defaultdict(asyncio.Lock)

    async def _send_upstream(
        self,
        *,
        method: str,
        upstream_url: str,
        headers: dict[str, str],
        payload: Optional[dict[str, Any]],
        proxy: Optional[str],
        timeout_value: int,
    ) -> httpx.Response:
        async with app.state.client_manager.get_client(upstream_url, proxy) as client:
            if method == "GET":
                return await client.get(upstream_url, headers=headers, timeout=timeout_value)
            if method == "POST":
                json_payload = await asyncio.to_thread(json.dumps, payload or {})
                return await client.post(upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
            if method == "PUT":
                json_payload = await asyncio.to_thread(json.dumps, payload or {})
                return await client.put(upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
        raise HTTPException(status_code=405, detail=f"Unsupported method: {method}")

    def _raw_response(self, upstream_resp: httpx.Response) -> Response:
        response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=response_headers,
            media_type=response_headers.get("content-type", "application/json"),
        )

    async def request_openapi(
        self,
        http_request: Request,
        request_body: Optional[dict[str, Any]],
        api_index: int,
        background_tasks: BackgroundTasks,
        *,
        method: str,
        openapi_path: str,
        endpoint: str = LINGJING_OPENAPI_ENDPOINT_PREFIX,
    ) -> Response:
        method_upper = method.upper()
        payload = request_body if isinstance(request_body, dict) else None
        request_model_name = _lingjing_request_model_for_openapi(payload, http_request.query_params)
        if not request_model_name:
            raise HTTPException(status_code=422, detail="Request requires a model")

        config = app.state.config
        current_info = request_info.get()
        current_info["model"] = current_info.get("model") or request_model_name
        request_id = _responses_request_id(current_info)

        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=endpoint,
            debug=is_debug,
            provider_resolver=get_right_order_providers,
        )
        _record_plan_observability(current_info, plan)
        runner = UpstreamRunner(
            plan,
            endpoint=endpoint,
            debug=is_debug,
        )
        last_error_response: dict[str, Any] = {}

        async def prepare_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            attempt.state["failure_stage"] = "validation"
            if not _is_lingjing_provider(provider):
                raise HTTPException(
                    status_code=400,
                    detail=f"{endpoint} only supports Lingjing providers",
                )

            upstream_url = _normalize_lingjing_openapi_upstream_url(
                provider.get("base_url", ""),
                openapi_path,
                query=_lingjing_upstream_query(http_request.url.query),
            )
            if not upstream_url:
                raise HTTPException(status_code=400, detail=f"{endpoint} requires provider base_url")

            proxy = safe_get(config, "preferences", "proxy", default=None)
            proxy = safe_get(provider, "preferences", "proxy", default=proxy)
            channel_id = f"{provider_name}"

            attempt.state["upstream_url"] = upstream_url
            attempt.state["channel_id"] = channel_id
            attempt.state["proxy"] = proxy
            attempt.state["failure_stage"] = "auth"
            attempt.provider_api_key_raw = await runner.select_provider_api_key(attempt)

            timeout_value = get_preference(
                app.state.provider_timeouts,
                provider_name,
                (original_model, request_model_name),
                DEFAULT_TIMEOUT,
            )
            attempt.state["api_key"] = attempt.provider_api_key_raw
            attempt.state["timeout_value"] = int(timeout_value)

        async def execute_attempt(attempt):
            provider = attempt.provider
            provider_name = attempt.provider_name
            original_model = attempt.original_model
            proxy = attempt.state["proxy"]
            api_key = attempt.state["api_key"]
            timeout_value = attempt.state["timeout_value"]
            upstream_url = attempt.state["upstream_url"]
            channel_id = attempt.state["channel_id"]
            headers = _lingjing_headers(provider, api_key, include_content_type=method_upper in {"POST", "PUT"})
            outbound_payload = payload
            if method_upper == "POST" and str(openapi_path or "").strip("/") == "draw/task/submit" and isinstance(payload, dict):
                outbound_payload = dict(payload)
                model_code = str(outbound_payload.get("modelCode") or "").strip()
                if not model_code or model_code == request_model_name:
                    outbound_payload["modelCode"] = original_model
                outbound_payload.pop("model", None)
                outbound_payload.pop("request_model", None)

            trace_logger.info(
                "endpoint=%s method=%s request_id=%s provider=%-11s model=%-22s engine=%-13s role=%s upstream_url=%s",
                endpoint,
                method_upper,
                request_id,
                channel_id[:11],
                request_model_name,
                "lingjing",
                plan.role,
                upstream_url,
            )
            attempt.state["failure_stage"] = "upstream"
            attempt.state["track_channel_stats"] = True

            _log_debug_request_headers(
                "DEBUG upstream request headers",
                headers,
                endpoint=endpoint,
                upstream_url=upstream_url,
                provider=channel_id,
                model=request_model_name,
                actual_model=original_model,
            )
            if outbound_payload is not None:
                _log_debug_request_body(
                    "DEBUG upstream request body",
                    outbound_payload,
                    endpoint=endpoint,
                    upstream_url=upstream_url,
                    provider=channel_id,
                    model=request_model_name,
                    actual_model=original_model,
                )

            upstream_resp = await self._send_upstream(
                method=method_upper,
                upstream_url=upstream_url,
                headers=headers,
                payload=outbound_payload,
                proxy=proxy,
                timeout_value=timeout_value,
            )
            success = 200 <= upstream_resp.status_code < 300
            current_info["first_response_time"] = 0 if success else -1
            current_info["success"] = success
            current_info["provider"] = channel_id if success else None
            background_tasks.add_task(
                update_channel_stats,
                current_info["request_id"],
                channel_id,
                request_model_name,
                current_info["api_key"],
                success=success,
                provider_api_key=attempt.provider_api_key_raw,
            )

            if not success:
                if 400 <= upstream_resp.status_code < 500 and upstream_resp.status_code not in (408, 409, 425, 429):
                    return self._raw_response(upstream_resp)
                last_error_response.clear()
                last_error_response.update(
                    {
                        "body": upstream_resp.content,
                        "headers": _copy_upstream_response_headers(upstream_resp.headers),
                    }
                )
                raise HTTPException(
                    status_code=upstream_resp.status_code,
                    detail=upstream_resp.content.decode("utf-8", errors="replace"),
                )

            return self._raw_response(upstream_resp)

        def after_failure(attempt, exc, status_code, error_message):
            if attempt.state.get("track_channel_stats"):
                background_tasks.add_task(
                    update_channel_stats,
                    current_info["request_id"],
                    attempt.state["channel_id"],
                    request_model_name,
                    current_info["api_key"],
                    success=False,
                    provider_api_key=attempt.provider_api_key_raw,
                )
            trace_logger.error(
                "%s upstream error status=%s error_type=%s request_id=%s model=%s provider=%s key=%s upstream_url=%s: %s",
                endpoint,
                status_code,
                type(exc).__name__,
                request_id,
                request_model_name,
                attempt.state.get("channel_id", attempt.provider_name),
                _mask_secret_for_log(attempt.provider_api_key_raw),
                attempt.state.get("upstream_url", ""),
                error_message,
            )

        def should_cool_down(exc, status_code, error_message, attempt):
            _ = exc, error_message, attempt
            return status_code in (401, 403, 429) or status_code >= 500

        def build_error_response(status_code, error_message):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            if last_error_response.get("body") is not None:
                headers = last_error_response.get("headers") or {}
                return Response(
                    content=last_error_response["body"],
                    status_code=status_code,
                    headers=headers,
                    media_type=headers.get("content-type", "application/json"),
                )
            return build_upstream_error_response(
                status_code=status_code,
                error_message=error_message,
                fallback_prefix="Error: Current provider response failed",
            )

        def build_final_response(completed_plan):
            current_info["first_response_time"] = -1
            current_info["success"] = False
            current_info["provider"] = None
            return JSONResponse(
                status_code=completed_plan.status_code,
                content={"error": f"All {request_model_name} error: {completed_plan.error_message}"},
            )

        return await runner.run(
            execute_attempt,
            prepare_attempt=prepare_attempt,
            after_failure=after_failure,
            build_error_response=build_error_response,
            build_final_response=build_final_response,
            should_cool_down=should_cool_down,
            on_retry=_record_retry_observability,
            on_cooldown=_record_cooldown_observability,
        )

model_handler = ModelRequestHandler()
responses_handler = ResponsesRequestHandler()
messages_handler = MessagesPassthroughHandler()
video_task_handler = VideoTaskHandler()
lingjing_openapi_handler = LingjingOpenapiHandler()

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = get_runtime_api_list()
    token = credentials.credentials
    api_index = None
    try:
        api_index = api_list.index(token)
    except ValueError:
        # 如果 token 不在 api_list 中，检查是否以 api_list 中的任何一个开头
        # api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
        api_index = None
    if api_index is None:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_index

async def verify_admin_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = get_runtime_api_list()
    token = credentials.credentials
    api_index = None
    try:
        api_index = api_list.index(token)
    except ValueError:
        # 如果 token 不在 api_list 中，检查是否以 api_list 中的任何一个开头
        # api_index = next((i for i, api in enumerate(api_list) if token.startswith(api)), None)
        api_index = None
    if api_index is None:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    # for api_key in app.state.api_keys_db:
    #     if token.startswith(api_key['api']):
    if len(api_list) == 1:
        return token
    if "admin" not in app.state.api_keys_db[api_index].get('role', ''):
        raise HTTPException(status_code=403, detail="Permission denied")
    return token

@app.get("/search", dependencies=[Depends(rate_limit_dependency)])
@app.get("/v1/search", dependencies=[Depends(rate_limit_dependency)])
async def jina_search(
    request: Request,
    background_tasks: BackgroundTasks,
    q: str = Query("Jina+AI"),
    api_index: int = Depends(verify_api_key),
):
    """
    Config-driven search routed through the existing provider selection/rotation architecture.

    Usage:
      - Provider config must include model: search (e.g. provider: jina + model: [search, ...])
      - User api key must include a rule like: jina/search
    """
    search_request = RequestModel(
        model="search",
        messages=[{"role": "user", "content": q}],
        stream=False,
    )
    return await model_handler.request_model(search_request, api_index, background_tasks, endpoint=str(request.url.path))

@app.post("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
async def chat_completions_route(request: RequestModel, background_tasks: BackgroundTasks, api_index: int = Depends(verify_api_key)):
    return await model_handler.request_model(request, api_index, background_tasks)

@app.post("/v1/responses", dependencies=[Depends(rate_limit_dependency)])
async def responses_route(
    http_request: Request,
    request: ResponsesRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    return await responses_handler.request_responses(http_request, request, api_index, background_tasks)

@app.post("/v1/responses/compact", dependencies=[Depends(rate_limit_dependency)])
async def responses_compact_route(
    http_request: Request,
    request: ResponsesRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    response = await responses_handler.request_responses(
        http_request,
        request,
        api_index,
        background_tasks,
        endpoint="/v1/responses/compact",
    )
    # response_body = getattr(response, "body", None)
    # if response_body is not None:
    #     print("print /v1/responses/compact:", response_body.decode("utf-8", errors="replace"))
    return response

@app.post("/v1/messages", dependencies=[Depends(rate_limit_dependency)])
async def messages_route(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request: dict[str, Any] = Body(...),
    api_index: int = Depends(verify_api_key),
):
    return await messages_handler.request_messages(
        http_request,
        request,
        api_index,
        background_tasks,
    )

# @app.options("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
# async def options_handler():
#     return JSONResponse(status_code=200, content={"detail": "OPTIONS allowed"})

@app.get("/v1/models", dependencies=[Depends(rate_limit_dependency)])
async def list_models(api_index: int = Depends(verify_api_key)):
    runtime_api_list = get_runtime_api_list()
    api_key = safe_get(runtime_api_list, api_index, default=None)
    model_response_cache = getattr(app.state, "model_response_cache", {}) or {}
    models = model_response_cache.get(api_key)
    if models is None:
        models = post_all_models(api_index, app.state.config, runtime_api_list, app.state.models_list)
    return JSONResponse(content={
        "object": "list",
        "data": models
    })

@app.post("/v1/images/generations", dependencies=[Depends(rate_limit_dependency)])
async def images_generations(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/images/generations")

@app.post("/v1/video/tasks", dependencies=[Depends(rate_limit_dependency)])
async def video_tasks_create(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request_body: dict[str, Any] = Body(...),
    api_index: int = Depends(verify_api_key),
):
    return await video_task_handler.create_task(
        http_request,
        request_body,
        api_index,
        background_tasks,
    )

@app.get("/v1/video/tasks/{task_id}", dependencies=[Depends(rate_limit_dependency)])
async def video_tasks_get(
    http_request: Request,
    task_id: str,
    background_tasks: BackgroundTasks,
    model: Optional[str] = Query(None),
    api_index: int = Depends(verify_api_key),
):
    return await video_task_handler.get_or_delete_task(
        http_request,
        task_id,
        api_index,
        background_tasks,
        method="GET",
        model=model,
    )

@app.post("/v1/asset-groups", dependencies=[Depends(rate_limit_dependency)])
async def asset_groups_create(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request_body: dict[str, Any] = Body(...),
    api_index: int = Depends(verify_api_key),
):
    return await lingjing_openapi_handler.request_openapi(
        http_request,
        request_body,
        api_index,
        background_tasks,
        method="POST",
        openapi_path="/material/asset-groups",
        endpoint=VIDEO_ASSET_GROUPS_ENDPOINT,
    )

@app.get("/v1/asset-groups/{group_id}", dependencies=[Depends(rate_limit_dependency)])
async def asset_group_get(
    http_request: Request,
    group_id: str,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    return await lingjing_openapi_handler.request_openapi(
        http_request,
        None,
        api_index,
        background_tasks,
        method="GET",
        openapi_path=f"/material/asset-groups/{quote(group_id, safe='')}",
        endpoint=VIDEO_ASSET_GROUPS_ENDPOINT,
    )

@app.post("/v1/assets", dependencies=[Depends(rate_limit_dependency)])
async def assets_create(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request_body: dict[str, Any] = Body(...),
    api_index: int = Depends(verify_api_key),
):
    return await lingjing_openapi_handler.request_openapi(
        http_request,
        request_body,
        api_index,
        background_tasks,
        method="POST",
        openapi_path="/material/assets/create",
        endpoint=VIDEO_ASSETS_ENDPOINT,
    )

@app.get("/v1/assets/{asset_id}", dependencies=[Depends(rate_limit_dependency)])
async def asset_get(
    http_request: Request,
    asset_id: str,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    return await lingjing_openapi_handler.request_openapi(
        http_request,
        None,
        api_index,
        background_tasks,
        method="GET",
        openapi_path=f"/material/assets/{quote(asset_id, safe='')}",
        endpoint=VIDEO_ASSETS_ENDPOINT,
    )

def _is_form_upload(value: Any) -> bool:
    return hasattr(value, "filename") and hasattr(value, "file")

def _form_text(value: Any) -> Optional[str]:
    if value is None or _is_form_upload(value):
        return None
    text = str(value).strip()
    return text or None

def _form_bool(value: Any, default: bool = False) -> bool:
    text = _form_text(value)
    if text is None:
        return default
    return text.lower() in ("1", "true", "yes", "on")

async def _parse_image_edit_request(http_request: Request) -> ImageEditRequest:
    content_type = (http_request.headers.get("content-type") or "").strip().lower()
    if content_type.startswith("application/json"):
        try:
            body = await http_request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        request = ImageEditRequest(**body)
        request.request_type = "image"
        return request

    if content_type and not content_type.startswith("multipart/form-data"):
        raise HTTPException(status_code=400, detail=f"Unsupported Content-Type {content_type!r}")

    try:
        form = await http_request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid multipart form: {exc}") from exc

    prompt = _form_text(form.get("prompt"))
    if prompt is None:
        raise HTTPException(status_code=400, detail="prompt is required")

    model = _form_text(form.get("model")) or "gpt-image-2"
    multipart_data: list[tuple[str, Any]] = []
    multipart_files: list[tuple[str, Any]] = []
    form_items = form.multi_items() if hasattr(form, "multi_items") else (
        (key, value) for key in form.keys() for value in form.getlist(key)
    )
    for key, value in form_items:
        if _is_form_upload(value):
            try:
                file_content = await value.read()
            finally:
                try:
                    await value.close()
                except Exception:
                    pass
            multipart_files.append((
                key,
                (
                    value.filename or "upload",
                    file_content,
                    value.content_type or "application/octet-stream",
                ),
            ))
        else:
            multipart_data.append((key, str(value)))

    request = ImageEditRequest(
        prompt=prompt,
        model=model,
        stream=_form_bool(form.get("stream"), False),
        multipart_data=multipart_data,
        multipart_files=multipart_files,
    )
    request.request_type = "image"
    return request

@app.post("/v1/images/edits", dependencies=[Depends(rate_limit_dependency)])
async def images_edits(
    http_request: Request,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    request = await _parse_image_edit_request(http_request)
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/images/edits")

@app.post("/v1/embeddings", dependencies=[Depends(rate_limit_dependency)])
async def embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/embeddings")

@app.post("/v1/audio/speech", dependencies=[Depends(rate_limit_dependency)])
async def audio_speech(
    request: TextToSpeechRequest,
    background_tasks: BackgroundTasks,
    api_index: str = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/audio/speech")

@app.post("/v1/moderations", dependencies=[Depends(rate_limit_dependency)])
async def moderations(
    request: ModerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/moderations")

@app.post("/v1/audio/transcriptions", dependencies=[Depends(rate_limit_dependency)])
async def audio_transcriptions(
    http_request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    api_index: int = Depends(verify_api_key)
):
    try:
        # Manually parse form data
        form_data = await http_request.form()
        # Use getlist to handle multiple values for the same key
        timestamp_granularities = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities: # If list is empty (parameter not sent)
            timestamp_granularities = None # Set to None to match Optional[List[str]]

        # 读取上传的文件内容 (file is still handled by FastAPI)
        content = await file.read()
        file_obj = io.BytesIO(content)

        # 创建AudioTranscriptionRequest对象
        request_obj = AudioTranscriptionRequest(
            file=(file.filename, file_obj, file.content_type),
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities
        )

        return await model_handler.request_model(request_obj, api_index, background_tasks, endpoint="/v1/audio/transcriptions")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid audio file encoding")
    except Exception as e:
        if is_debug:
            import traceback
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")

@app.get("/v1/generate-api-key", dependencies=[Depends(rate_limit_dependency)])
async def generate_api_key():
    # Define the character set (only alphanumeric)
    chars = string.ascii_letters + string.digits
    # Generate a random string of 36 characters
    random_string = ''.join(secrets.choice(chars) for _ in range(48))
    api_key = "sk-" + random_string
    return JSONResponse(content={"api_key": api_key})

# 在 /stats 路由中返回成功和失败百分比
@app.get("/v1/stats", dependencies=[Depends(rate_limit_dependency)])
async def get_stats(
    request: Request,
    token: str = Depends(verify_admin_api_key),
    hours: int = Query(default=24, ge=1, le=720, description="Number of hours to look back for stats (1-720)")
):
    '''
    ## 获取统计数据

    使用 `/v1/stats` 获取最近 24 小时各个渠道的使用情况统计。同时带上 自己的 uni-api 的 admin API key。

    数据包括：

    1. 每个渠道下面每个模型的成功率，成功率从高到低排序。
    2. 每个渠道总的成功率，成功率从高到低排序。
    3. 每个模型在所有渠道总的请求次数。
    4. 每个端点的请求次数。
    5. 每个ip请求的次数。

    `/v1/stats?hours=48` 参数 `hours` 可以控制返回最近多少小时的数据统计，不传 `hours` 这个参数，默认统计最近 24 小时的统计数据。

    还有其他统计数据，可以自己写sql在数据库自己查。其他数据包括：首字时间，每个请求的总处理时间，每次请求是否成功，每次请求是否符合道德审查，每次请求的文本内容，每次请求的 API key，每次请求的输入 token，输出 token 数量。
    '''
    if DISABLE_DATABASE:
        return JSONResponse(content={"stats": {}})
    async with async_session() as session:
        # 计算指定时间范围的开始时间
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # 1. 每个渠道下面每个模型的成功率
        channel_model_stats = await session.execute(
            select(
                ChannelStat.provider,
                ChannelStat.model,
                func.count().label('total'),
                func.sum(case((ChannelStat.success, 1), else_=0)).label('success_count')
            )
            .where(ChannelStat.timestamp >= start_time)
            .group_by(ChannelStat.provider, ChannelStat.model)
        )
        channel_model_stats = channel_model_stats.fetchall()

        # 2. 每个渠道总的成功率
        channel_stats = await session.execute(
            select(
                ChannelStat.provider,
                func.count().label('total'),
                func.sum(case((ChannelStat.success, 1), else_=0)).label('success_count')
            )
            .where(ChannelStat.timestamp >= start_time)
            .group_by(ChannelStat.provider)
        )
        channel_stats = channel_stats.fetchall()

        # 3. 每个模型在所有渠道总的请求次数
        model_stats = await session.execute(
            select(RequestStat.model, func.count().label('count'))
            .where(RequestStat.timestamp >= start_time)
            .group_by(RequestStat.model)
            .order_by(desc('count'))
        )
        model_stats = model_stats.fetchall()

        # 4. 每个端点的请求次数
        endpoint_stats = await session.execute(
            select(RequestStat.endpoint, func.count().label('count'))
            .where(RequestStat.timestamp >= start_time)
            .group_by(RequestStat.endpoint)
            .order_by(desc('count'))
        )
        endpoint_stats = endpoint_stats.fetchall()

        # 5. 每个ip请求的次数
        ip_stats = await session.execute(
            select(RequestStat.client_ip, func.count().label('count'))
            .where(RequestStat.timestamp >= start_time)
            .group_by(RequestStat.client_ip)
            .order_by(desc('count'))
        )
        ip_stats = ip_stats.fetchall()

    # 处理统计数据并返回
    stats = {
        "time_range": f"Last {hours} hours",
        "channel_model_success_rates": [
            {
                "provider": stat.provider,
                "model": stat.model,
                "success_rate": stat.success_count / stat.total if stat.total > 0 else 0,
                "total_requests": stat.total
            } for stat in sorted(channel_model_stats, key=lambda x: x.success_count / x.total if x.total > 0 else 0, reverse=True)
        ],
        "channel_success_rates": [
            {
                "provider": stat.provider,
                "success_rate": stat.success_count / stat.total if stat.total > 0 else 0,
                "total_requests": stat.total
            } for stat in sorted(channel_stats, key=lambda x: x.success_count / x.total if x.total > 0 else 0, reverse=True)
        ],
        "model_request_counts": [
            {
                "model": stat.model,
                "count": stat.count
            } for stat in model_stats
        ],
        "endpoint_request_counts": [
            {
                "endpoint": stat.endpoint,
                "count": stat.count
            } for stat in endpoint_stats
        ],
        "ip_request_counts": [
            {
                "ip": stat.client_ip,
                "count": stat.count
            } for stat in ip_stats
        ]
    }

    return JSONResponse(content=stats)

@app.get("/", dependencies=[Depends(rate_limit_dependency)])
async def root():
    return RedirectResponse(url="https://uni-api-web.pages.dev", status_code=302)

# async def on_fetch(request, env):
#     import asgi
#     return await asgi.fetch(app, request, env)

@app.get("/v1/api_config", dependencies=[Depends(rate_limit_dependency)])
async def api_config(api_index: int = Depends(verify_admin_api_key)):
    encoded_config = jsonable_encoder(app.state.config)
    return JSONResponse(content={"api_config": encoded_config})

@app.post("/v1/api_config/update", dependencies=[Depends(rate_limit_dependency)])
async def api_config_update(api_index: int = Depends(verify_admin_api_key), config: dict = Body(...)):
    if "providers" in config:
        app.state.config["providers"] = config["providers"]
        app.state.config, app.state.api_keys_db, app.state.api_list = await update_config(
            app.state.config,
            use_config_url=False,
        )
        await refresh_runtime_state(app)
    return JSONResponse(content={"message": "API config updated"})

# Pydantic Models for Token Usage Response
class TokenUsageEntry(BaseModel):
    api_key_prefix: str
    model: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    request_count: int

class QueryDetails(BaseModel):
    model_config = {'protected_namespaces': ()}

    start_datetime: Optional[str] = None # e.g., "2023-10-27T10:00:00Z" or Unix timestamp
    end_datetime: Optional[str] = None   # e.g., "2023-10-28T12:30:45Z" or Unix timestamp
    api_key_filter: Optional[str] = None
    model_filter: Optional[str] = None
    credits: Optional[str] = None
    total_cost: Optional[str] = None
    balance: Optional[str] = None

class TokenUsageResponse(BaseModel):
    usage: List[TokenUsageEntry]
    query_details: QueryDetails


class ChannelKeyRanking(BaseModel):
    api_key: str
    success_count: int
    total_requests: int
    success_rate: float


class ChannelKeyRankingsResponse(BaseModel):
    rankings: List[ChannelKeyRanking]
    query_details: QueryDetails


async def query_token_usage(
    session: AsyncSession,
    filter_api_key: Optional[str] = None,
    filter_model: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None
) -> List[Dict]:
    """Queries the RequestStat table for aggregated token usage."""
    query = select(
        RequestStat.api_key,
        RequestStat.model,
        func.sum(RequestStat.prompt_tokens).label("total_prompt_tokens"),
        func.sum(RequestStat.completion_tokens).label("total_completion_tokens"),
        func.sum(RequestStat.total_tokens).label("total_tokens"),
        func.count(RequestStat.id).label("request_count")
    ).group_by(RequestStat.api_key, RequestStat.model)

    # Apply filters
    if filter_api_key:
        query = query.where(RequestStat.api_key == filter_api_key)
    if filter_model:
        query = query.where(RequestStat.model == filter_model)
    if start_dt:
        query = query.where(RequestStat.timestamp >= start_dt)
    if end_dt:
        # Make end_dt inclusive by adding one day
        query = query.where(RequestStat.timestamp < end_dt + timedelta(days=1))

    # Filter out entries with null or empty model if not specifically requested
    if not filter_model:
         query = query.where(RequestStat.model.isnot(None) & (RequestStat.model != ''))


    result = await session.execute(query)
    rows = result.mappings().all()

    # Process results: mask API key
    processed_usage = []
    for row in rows:
        usage_dict = dict(row)
        api_key = usage_dict.get("api_key", "")
        # Mask API key (show prefix like sk-...xyz)
        if api_key and len(api_key) > 7:
            prefix = api_key[:7]
            suffix = api_key[-4:]
            usage_dict["api_key_prefix"] = f"{prefix}...{suffix}"
        else:
            usage_dict["api_key_prefix"] = api_key # Show short keys as is or handle None
        del usage_dict["api_key"] # Remove original full key
        processed_usage.append(usage_dict)

    return processed_usage

async def get_usage_data(filter_api_key: Optional[str] = None, filter_model: Optional[str] = None,
                        start_dt_obj: Optional[datetime] = None, end_dt_obj: Optional[datetime] = None) -> List[Dict]:
    """
    查询数据库并获取令牌使用数据。
    这个函数封装了创建会话和查询令牌使用情况的逻辑。

    Args:
        filter_api_key: 可选的API密钥过滤器
        filter_model: 可选的模型过滤器
        start_dt_obj: 开始日期时间
        end_dt_obj: 结束日期时间

    Returns:
        包含令牌使用统计数据的列表
    """
    async with async_session() as session:
        usage_data = await query_token_usage(
            session=session,
            filter_api_key=filter_api_key,
            filter_model=filter_model,
            start_dt=start_dt_obj,
            end_dt=end_dt_obj
        )
    return usage_data

@app.get("/v1/token_usage", response_model=TokenUsageResponse, dependencies=[Depends(rate_limit_dependency)])
async def get_token_usage(
    request: Request, # Inject request to access app.state
    api_key_param: Optional[str] = None, # Query param for admin filtering
    model: Optional[str] = None,
    start_datetime: Optional[str] = None, # ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp
    end_datetime: Optional[str] = None,   # ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp
    last_n_days: Optional[int] = None,
    api_index: tuple = Depends(verify_api_key) # Use verify_api_key for auth and getting token/index
):
    """
    Retrieves aggregated token usage statistics based on API key and model,
    filtered by a specified time range.
    Admin users can filter by specific API keys.
    """
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")

    requesting_token = safe_get(app.state.config, 'api_keys', api_index, "api", default="") # verify_api_key returns the token directly now

    # Determine admin status
    is_admin = False
    # print("app.state.admin_api_key", app.state.admin_api_key, requesting_token, requesting_token in app.state.admin_api_key)
    if hasattr(app.state, "admin_api_key") and requesting_token in app.state.admin_api_key:
        is_admin = True

    # Determine API key filter
    filter_api_key = None
    api_key_filter_detail = "all" # For response details
    # print("api_key_param", is_admin, api_key_param)
    if is_admin:
        if api_key_param:
            filter_api_key = api_key_param
            api_key_filter_detail = api_key_param
        # else: filter_api_key remains None (all users)
    else:
        # Non-admin can only see their own stats
        filter_api_key = requesting_token
        api_key_filter_detail = "self"

    # Determine time range
    end_dt_obj = None
    start_dt_obj = None
    start_datetime_detail = None
    end_datetime_detail = None

    now = datetime.now(timezone.utc)

    def parse_datetime_input(dt_input: str) -> datetime:
        """Parses ISO 8601 string or Unix timestamp."""
        try:
            # Try parsing as Unix timestamp first
            return datetime.fromtimestamp(float(dt_input), tz=timezone.utc)
        except ValueError:
            # Try parsing as ISO 8601 format
            try:
                # Handle potential 'Z' for UTC timezone explicitly
                if dt_input.endswith('Z'):
                    dt_input = dt_input[:-1] + '+00:00'
                # Use fromisoformat for robust parsing
                dt_obj = datetime.fromisoformat(dt_input)
                # Ensure timezone is UTC if naive
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                # Convert to UTC if it has another timezone
                return dt_obj.astimezone(timezone.utc)
            except ValueError:
                raise ValueError(f"Invalid datetime format: {dt_input}. Use ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp.")


    if last_n_days is not None:
        if start_datetime or end_datetime:
            raise HTTPException(status_code=400, detail="Cannot use last_n_days with start_datetime or end_datetime.")
        if last_n_days <= 0:
            raise HTTPException(status_code=400, detail="last_n_days must be positive.")
        start_dt_obj = now - timedelta(days=last_n_days)
        end_dt_obj = now # Use current time as end for last_n_days
        start_datetime_detail = start_dt_obj.isoformat(timespec='seconds')
        end_datetime_detail = end_dt_obj.isoformat(timespec='seconds')
    elif start_datetime or end_datetime:
        try:
            if start_datetime:
                start_dt_obj = parse_datetime_input(start_datetime)
                start_datetime_detail = start_dt_obj.isoformat(timespec='seconds')
            if end_datetime:
                end_dt_obj = parse_datetime_input(end_datetime)
                end_datetime_detail = end_dt_obj.isoformat(timespec='seconds')
            # Basic validation: end datetime should not be before start datetime
            if start_dt_obj and end_dt_obj and end_dt_obj < start_dt_obj:
                 raise HTTPException(status_code=400, detail="end_datetime cannot be before start_datetime.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Default to last 30 days if no range specified
        start_dt_obj = now - timedelta(days=30)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec='seconds')
        end_datetime_detail = end_dt_obj.isoformat(timespec='seconds')

    # 使用新的get_usage_data函数替代直接的数据库查询代码
    usage_data = await get_usage_data(
        filter_api_key=filter_api_key,
        filter_model=model,
        start_dt_obj=start_dt_obj,
        end_dt_obj=end_dt_obj
    )
    # print("usage_data", usage_data)

    if filter_api_key:
        credits, total_cost = await update_paid_api_keys_states(app, filter_api_key)
    else:
        credits, total_cost = None, None

    # Prepare response
    query_details = QueryDetails(
        start_datetime=start_datetime_detail,
        end_datetime=end_datetime_detail,
        api_key_filter=api_key_filter_detail,
        model_filter=model if model else "all",
        credits= "$" + str(credits),
        total_cost= "$" + str(total_cost),
        balance= "$" + str(float(credits) - float(total_cost)) if credits and total_cost else None
    )

    response_data = TokenUsageResponse(
        usage=[TokenUsageEntry(**item) for item in usage_data],
        query_details=query_details
    )

    return response_data


@app.get(
    "/v1/channel_key_rankings",
    response_model=ChannelKeyRankingsResponse,
    dependencies=[Depends(rate_limit_dependency)],
)
async def get_channel_key_rankings(
    request: Request,
    provider_name: str,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    last_n_days: Optional[int] = None,
    token: str = Depends(verify_admin_api_key),
):
    """
    Retrieves the success rate ranking of API keys for a specific channel,
    filtered by a specified time range.
    """
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")

    end_dt_obj = None
    start_dt_obj = None
    start_datetime_detail = None
    end_datetime_detail = None

    now = datetime.now(timezone.utc)

    def parse_datetime_input(dt_input: str) -> datetime:
        """Parses ISO 8601 string or Unix timestamp."""
        try:
            return datetime.fromtimestamp(float(dt_input), tz=timezone.utc)
        except ValueError:
            try:
                if dt_input.endswith("Z"):
                    dt_input = dt_input[:-1] + "+00:00"
                dt_obj = datetime.fromisoformat(dt_input)
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                return dt_obj.astimezone(timezone.utc)
            except ValueError:
                raise ValueError(
                    f"Invalid datetime format: {dt_input}. Use ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp."
                )

    if last_n_days is not None:
        if start_datetime or end_datetime:
            raise HTTPException(
                status_code=400,
                detail="Cannot use last_n_days with start_datetime or end_datetime.",
            )
        if last_n_days <= 0:
            raise HTTPException(
                status_code=400, detail="last_n_days must be positive."
            )
        start_dt_obj = now - timedelta(days=last_n_days)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
        end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")
    elif start_datetime or end_datetime:
        try:
            if start_datetime:
                start_dt_obj = parse_datetime_input(start_datetime)
                start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
            if end_datetime:
                end_dt_obj = parse_datetime_input(end_datetime)
                end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")
            if start_dt_obj and end_dt_obj and end_dt_obj < start_dt_obj:
                raise HTTPException(
                    status_code=400, detail="end_datetime cannot be before start_datetime."
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Default to last 24 hours if no range specified
        start_dt_obj = now - timedelta(days=1)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
        end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")

    rankings_data = await query_channel_key_stats(
        provider_name=provider_name, start_dt=start_dt_obj, end_dt=end_dt_obj
    )

    query_details = QueryDetails(
        start_datetime=start_datetime_detail,
        end_datetime=end_datetime_detail,
        api_key_filter=provider_name,
    )

    response_data = ChannelKeyRankingsResponse(
        rankings=[ChannelKeyRanking(**item) for item in rankings_data],
        query_details=query_details,
    )

    return response_data


class TokenInfo(BaseModel):
    api_key_prefix: str
    model: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    request_count: int

class ApiKeyState(BaseModel):
    credits: float
    created_at: datetime
    all_tokens_info: List[Dict[str, Any]]
    total_cost: float
    enabled: bool

    @field_serializer('created_at')
    def serialize_dt(self, dt: datetime):
        return dt.isoformat()

class ApiKeysStatesResponse(BaseModel):
    api_keys_states: Dict[str, ApiKeyState]

@app.get("/v1/api_keys_states", dependencies=[Depends(rate_limit_dependency)])
async def api_keys_states(token: str = Depends(verify_admin_api_key)):
    # 转换原始状态数据为Pydantic模型
    states_dict = {}
    for key, state in app.state.paid_api_keys_states.items():
        # 创建ApiKeyState对象
        states_dict[key] = ApiKeyState(
            credits=state["credits"],
            created_at=state["created_at"],
            all_tokens_info=state["all_tokens_info"],
            total_cost=state["total_cost"],
            enabled=state["enabled"]
        )

    # 创建响应模型
    response = ApiKeysStatesResponse(api_keys_states=states_dict)

    # 返回JSON序列化结果
    return response

@app.post("/v1/add_credits", dependencies=[Depends(rate_limit_dependency)])
async def add_credits_to_api_key(
    request: Request, # Inject request to access app.state
    paid_key: str = Query(..., description="The API key to add credits to"),
    amount: float = Query(..., description="The amount of credits to add. Must be positive.", gt=0),
    token: str = Depends(verify_admin_api_key)
):
    if paid_key not in app.state.paid_api_keys_states:
        raise HTTPException(status_code=404, detail=f"API key '{paid_key}' not found in paid API keys states.")

    # The validation `amount > 0` is handled by `Query(..., gt=0)`

    # 更新 credits
    # Ensure 'amount' is treated as float, though Query should handle conversion.
    app.state.paid_api_keys_states[paid_key]["credits"] += float(amount)

    # 更新 enabled 状态
    current_credits = app.state.paid_api_keys_states[paid_key]["credits"]
    total_cost = app.state.paid_api_keys_states[paid_key]["total_cost"]
    app.state.paid_api_keys_states[paid_key]["enabled"] = current_credits >= total_cost

    logger.info(f"Credits for API key '{paid_key}' updated. Amount added: {amount}, New credits: {current_credits}, Enabled: {app.state.paid_api_keys_states[paid_key]['enabled']}")

    return JSONResponse(content={
        "message": f"Successfully added {amount} credits to API key '{paid_key}'.",
        "paid_key": paid_key,
        "new_credits": current_credits,
        "enabled": app.state.paid_api_keys_states[paid_key]["enabled"]
    })

# 添加静态文件挂载
app.mount("/", StaticFiles(directory="./static", html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    import os
    PORT = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        reload_dirs=["./"],
        reload_includes=["*.py", "api.yaml"],
        reload_excludes=["./data"],
        ws="none",
        # log_level="warning"
    )
