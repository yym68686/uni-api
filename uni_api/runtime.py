import os
import re
import json
import uuid
import codecs
import functools
from dataclasses import dataclass, field
import httpx
import string
import secrets
import tomllib
import asyncio
import random
from asyncio import Semaphore
from time import time
from urllib.parse import urlparse
from collections import defaultdict
from contextlib import aclosing, asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Dict, Union, Optional, List, Any, Awaitable, Callable
from pydantic import BaseModel, field_serializer

from starlette.responses import Response
from starlette.responses import StreamingResponse as StarletteStreamingResponse

from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Depends, Request, Body, BackgroundTasks, UploadFile, File, Form, Query

from core.log_config import logger, trace_logger
from uni_api.providers.payloads import (
    CODEX_CLI_VERSION,
    CODEX_USER_AGENT,
    apply_post_body_parameter_overrides,
    force_codex_client_headers,
    strip_unsupported_codex_payload_fields,
)
from uni_api.providers.header_passthrough import apply_provider_preference_headers
from uni_api.providers.responses import fetch_response, fetch_response_stream
from core.models import RequestModel, ResponsesRequest, ImageGenerationRequest, ImageEditRequest, AudioTranscriptionRequest, ModerationRequest, TextToSpeechRequest, EmbeddingRequest
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
from uni_api.routing.index import (
    build_api_key_model_response_cache,
    build_api_key_models_map,
    build_routing_index,
    estimate_request_total_tokens,
)
from uni_api.routing.planner import (
    RoutingPlan,
    _call_provider_resolver,
    get_right_order_providers,
    select_provider_api_key_raw,
)
from upstream import (
    UPSTREAM_NETWORK_ERRORS,
    UpstreamRunner,
    build_upstream_error_response,
)
from fugue_observability import (
    start_fugue_observability_from_env,
    stop_fugue_observability,
)
from uni_api.auth.api_keys import (
    extract_api_key_from_headers,
    require_admin_api_key,
    require_api_key_index,
    resolve_api_key_index,
)
from uni_api.auth.codex_oauth import (
    CodexOAuthTokenManager,
    CodexRefreshTokenStore,
    split_codex_api_key,
)
from uni_api.api.admin import api_config_response, api_config_update_response
from uni_api.api.chat import (
    chat_completions_response,
    messages_response,
    responses_api_response,
    search_response,
)
from uni_api.api.health import healthz_response, observability_runtime_response
from uni_api.api.media import (
    audio_speech_response,
    audio_transcription_response,
    embeddings_response,
    image_edit_response,
    image_generation_response,
    moderation_response,
)
from uni_api.api.models import list_models_payload
from uni_api.api.stats import (
    ApiKeysStatesResponse,
    ChannelKeyRankingsResponse,
    TokenUsageResponse,
    add_credits_response,
    api_keys_states_response,
    channel_key_rankings_response,
    stats_summary_response,
    token_usage_response,
)
from uni_api.api.video import (
    asset_get_response,
    asset_group_get_response,
    asset_groups_create_response,
    assets_create_response,
    video_task_create_response,
    video_task_get_response,
)
from uni_api.app_state import AppRuntimeSnapshot
import uni_api.config.legacy_loader as legacy_config_loader
from uni_api.config.compiler import compile_runtime_config
from uni_api.config.timeout_policy import apply_timeout_policy, init_timeout_policy
from uni_api.observability.paid_keys import compute_paid_api_key_state
from uni_api.observability.request_context import (
    get_request_info,
    request_info,
)
from uni_api.observability.spans import merge_timing_spans
from uni_api.observability.telemetry import emit_request_observability
from uni_api.observability.middleware import (
    StatsMiddleware,
    StatsMiddlewareDependencies,
)
from uni_api.middleware.request_decompression import RequestBodyDecompressionMiddleware
from uni_api.persistence.repositories import StatsRepository
from uni_api.providers import ProviderRegistry
from uni_api.providers.execution import prepare_provider_request
from uni_api.providers.adapters import default_provider_adapters
from uni_api.streaming.cleanup import (
    await_stream_cleanup_safely,
    background_stream_cleanup_snapshot,
    call_cleanup_safely,
    force_close_response_httpcore_stream_chain_safely,
    force_release_httpcore_pool_request_safely,
    wait_background_stream_cleanup_tasks,
)
from uni_api.streaming.logging_response import LoggingStreamingResponse
from uni_api.upstream.client_pool import ClientPool
from uni_api.upstream.urls import (
    lingjing_upstream_query,
    normalize_content_generation_tasks_upstream_url,
    normalize_lingjing_draw_task_upstream_url,
    normalize_lingjing_openapi_upstream_url,
    normalize_messages_upstream_url,
    normalize_responses_compact_upstream_url,
    normalize_responses_upstream_url,
)
from video import VideoAdapterError

from uni_api.api.models import post_all_models
from uni_api.config.legacy_loader import (
    load_config,
    update_config,
)
from uni_api.persistence.key_stats import get_sorted_api_keys, query_channel_key_stats
from uni_api.rate_limit.memory import InMemoryRateLimiter
from uni_api.upstream.error_handling import error_handling_wrapper
from core.utils import safe_get

from sqlalchemy import inspect, text
from sqlalchemy.sql import sqltypes
from sqlalchemy.ext.asyncio import AsyncSession

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


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
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
            elapsed_ms = int((time() - self.started_at) * 1000)
            if name != "request_received":
                elapsed_ms = max(1, elapsed_ms)
            self.spans[name] = elapsed_ms

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


def _coerce_request_trace(current_info: dict[str, Any]) -> Optional[RequestTrace]:
    if not isinstance(current_info, dict):
        return None
    trace = current_info.get("trace")
    if isinstance(trace, RequestTrace):
        spans = current_info.get("timing_spans")
        if isinstance(spans, dict):
            for key, value in spans.items():
                name = str(key or "").strip()
                if name and name not in trace.spans:
                    trace.spans[name] = int(value) if isinstance(value, float) else value
        merge_timing_spans(current_info, trace.snapshot())
        return trace
    trace_id = str(current_info.get("trace_id") or current_info.get("request_id") or "").strip()
    if not trace_id:
        return None
    trace = RequestTrace(
        trace_id=trace_id,
        parent_span_id=current_info.get("parent_span_id"),
        trace_flags=current_info.get("trace_flags"),
        tracestate=current_info.get("tracestate"),
    )
    spans = current_info.get("timing_spans")
    if isinstance(spans, dict):
        for key, value in spans.items():
            name = str(key or "").strip()
            if not name:
                continue
            if isinstance(value, (int, str)):
                trace.spans[name] = value
            elif isinstance(value, float):
                trace.spans[name] = int(value)
    current_info["trace"] = trace
    merge_timing_spans(current_info, trace.snapshot())
    return trace


def _fallback_stage_elapsed_ms(current_info: dict[str, Any], stage: str) -> int:
    if stage == "request_received":
        return 0
    start_time = current_info.get("start_time") if isinstance(current_info, dict) else None
    if isinstance(start_time, (int, float)):
        return max(1, int((time() - float(start_time)) * 1000))
    return 1


def _mark_current_info_stage(current_info: dict[str, Any], stage: str) -> None:
    name = str(stage or "").strip()
    if not isinstance(current_info, dict) or not name:
        return
    trace = _coerce_request_trace(current_info)
    if isinstance(trace, RequestTrace):
        trace.mark(name)
        merge_timing_spans(current_info, trace.snapshot())
        return
    spans = dict(current_info.get("timing_spans") or {})
    spans[name] = _fallback_stage_elapsed_ms(current_info, name)
    merge_timing_spans(current_info, spans)


def _set_current_info_trace_tag(current_info: dict[str, Any], name: str, value: Optional[str]) -> None:
    key = str(name or "").strip()
    text = str(value or "").strip()
    if not isinstance(current_info, dict) or not key or not text:
        return
    trace = _coerce_request_trace(current_info)
    if isinstance(trace, RequestTrace):
        trace.set_tag(key, text)
        merge_timing_spans(current_info, trace.snapshot())
        return
    spans = dict(current_info.get("timing_spans") or {})
    spans[key] = text[:128]
    merge_timing_spans(current_info, spans)


def _add_current_info_trace_ms(current_info: dict[str, Any], name: str, value_ms: Any) -> None:
    key = str(name or "").strip()
    if not isinstance(current_info, dict) or not key:
        return
    trace = _coerce_request_trace(current_info)
    if isinstance(trace, RequestTrace):
        trace.add_ms(key, value_ms)
        merge_timing_spans(current_info, trace.snapshot())
        return
    try:
        value = max(0, int(round(float(value_ms))))
    except (TypeError, ValueError):
        return
    spans = dict(current_info.get("timing_spans") or {})
    spans[key] = value
    merge_timing_spans(current_info, spans)


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
            info = get_request_info()
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
    return await await_stream_cleanup_safely(awaitable, label=label)


async def _call_cleanup_safely(cleanup: Callable[[], Any], *, label: str) -> bool:
    return await call_cleanup_safely(cleanup, label=label)


async def _force_release_httpcore_pool_request_safely(stream: Any) -> bool:
    return await force_release_httpcore_pool_request_safely(
        stream,
        label="Upstream HTTP pool request",
    )


async def _force_close_httpcore_stream_chain_safely(upstream_response: Any) -> bool:
    return await force_close_response_httpcore_stream_chain_safely(
        upstream_response,
        label="Upstream HTTP response stream",
    )


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
        info = get_request_info()
    except LookupError:
        return None
    trace = info.get("trace") if isinstance(info, dict) else None
    return trace if isinstance(trace, RequestTrace) else None


def _request_state_current_info(http_request: Optional[Request]) -> Optional[dict[str, Any]]:
    state = getattr(http_request, "state", None)
    info = getattr(state, "uni_api_request_info", None)
    return info if isinstance(info, dict) else None


def _mark_stage(stage: str) -> None:
    trace = _current_trace()
    if trace is not None:
        trace.mark(stage)
        try:
            info = get_request_info()
        except LookupError:
            return
        if isinstance(info, dict):
            merge_timing_spans(info, trace.snapshot())


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
    _mark_current_info_stage(current_info, "upstream_first_chunk")
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
    info = get_request_info()
    if not isinstance(info, dict):
        return
    retry_count = int(info.get("retry_count") or 0) + 1
    info["retry_count"] = retry_count
    info["error_type"] = type(error_message).__name__ if not isinstance(error_message, str) else "upstream_retry"
    _mark_current_info_stage(info, "retry_started")
    _add_current_info_trace_ms(info, "retry_count", retry_count)
    _add_current_info_trace_ms(info, "retry_status_code", status_code)
    _set_current_info_trace_tag(info, "retry_provider", getattr(attempt, "provider_name", None))
    _set_current_info_trace_tag(info, "retry_error_type", info.get("error_type"))


def _record_cooldown_observability(attempt: Any, status_code: int, error_message: Any) -> None:
    _ = error_message
    info = get_request_info()
    if not isinstance(info, dict):
        return
    cooldown_count = int(info.get("cooldown_count") or 0) + 1
    info["cooldown_count"] = cooldown_count
    _add_current_info_trace_ms(info, "cooldown_count", cooldown_count)
    _add_current_info_trace_ms(info, "cooldown_status_code", status_code)
    _set_current_info_trace_tag(info, "cooldown_provider", getattr(attempt, "provider_name", None))


def _emit_request_observability(current_info: dict[str, Any]) -> None:
    if not isinstance(current_info, dict) or current_info.get("_fugue_observability_emitted"):
        return
    current_info["_fugue_observability_emitted"] = True
    try:
        emit_request_observability(
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

SENSITIVE_DEBUG_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
}


def _debug_header_value(name: Any, value: Any) -> str:
    header_name = str(name or "").strip().lower()
    header_value = str(value)
    if header_name in SENSITIVE_DEBUG_HEADERS or "token" in header_name or "secret" in header_name:
        return _mask_secret_for_log(header_value)
    return header_value


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
            pairs.append({"name": str(key), "value": _debug_header_value(key, value)})
        return pairs

    if hasattr(headers, "multi_items"):
        return [
            {"name": str(key), "value": _debug_header_value(key, value)}
            for key, value in headers.multi_items()
        ]

    if hasattr(headers, "items"):
        return [
            {"name": str(key), "value": _debug_header_value(key, value)}
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

    if legacy_config_loader.yaml_error_message:
        raise HTTPException(
            status_code=500,
            detail={"error": legacy_config_loader.yaml_error_message},
        )
    raise HTTPException(
        status_code=500,
        detail={"error": "No API key found in api.yaml"},
    )


async def refresh_runtime_state(app: FastAPI) -> None:
    config = getattr(app.state, "config", {}) or {}
    api_keys_db = getattr(app.state, "api_keys_db", []) or []
    api_list = getattr(app.state, "api_list", []) or []

    models_list = build_api_key_models_map(config, api_list)
    runtime_config = compile_runtime_config(
        config,
        api_list,
        models_list=models_list,
        default_timeout=DEFAULT_TIMEOUT,
    )
    runtime_snapshot = AppRuntimeSnapshot(
        runtime_config=runtime_config,
        provider_registry=ProviderRegistry(default_provider_adapters()),
        user_api_keys_rate_limit=_build_user_api_keys_rate_limit(config, api_list),
        global_rate_limit=parse_rate_limit(
            safe_get(config, "preferences", "rate_limit", default="999999/min")
        ),
        admin_api_key=_build_admin_api_keys(api_keys_db),
        provider_timeouts=init_preference(config, "model_timeout", DEFAULT_TIMEOUT),
        timeout_policy=init_timeout_policy(config),
        keepalive_interval=init_preference(config, "keepalive_interval", 99999),
    )

    app.state.runtime_snapshot = runtime_snapshot
    app.state.runtime_config = runtime_snapshot.runtime_config
    app.state.provider_registry = runtime_snapshot.provider_registry
    app.state.user_api_keys_rate_limit = runtime_snapshot.user_api_keys_rate_limit
    app.state.global_rate_limit = runtime_snapshot.global_rate_limit
    app.state.admin_api_key = runtime_snapshot.admin_api_key
    app.state.provider_timeouts = runtime_snapshot.provider_timeouts
    app.state.timeout_policy = runtime_snapshot.timeout_policy
    app.state.keepalive_interval = runtime_snapshot.keepalive_interval
    app.state.models_list = runtime_config.api_key_allowed_models
    app.state.routing_index = runtime_config.routing_index
    app.state.model_response_cache = runtime_config.api_key_model_response_cache
    app.state.api_key_index = {api_key: index for index, api_key in enumerate(api_list)}
    app.state.runtime_config_source_id = id(config)

    if not DISABLE_DATABASE:
        app.state.paid_api_keys_states = {}
        for paid_key in api_list:
            await update_paid_api_keys_states(app, paid_key)


def _iter_provider_key_pools(app: FastAPI):
    seen: set[int] = set()
    for pool in list(provider_api_circular_list.values()):
        pool_id = id(pool)
        if pool_id not in seen:
            seen.add(pool_id)
            yield pool

    user_pools = getattr(app.state, "user_api_keys_rate_limit", {}) or {}
    values = user_pools.values() if hasattr(user_pools, "values") else []
    for pool in list(values):
        pool_id = id(pool)
        if pool_id not in seen:
            seen.add(pool_id)
            yield pool


def provider_key_pools_snapshot(app: FastAPI) -> dict[str, Any]:
    snapshots = [
        pool.snapshot()
        for pool in _iter_provider_key_pools(app)
        if hasattr(pool, "snapshot")
    ]
    return {
        "total": len(snapshots),
        "reordering_task_active": sum(1 for item in snapshots if item.get("reordering_task_active")),
        "reordering_task_done": sum(1 for item in snapshots if item.get("reordering_task_done")),
    }


async def close_provider_key_pools(app: FastAPI) -> dict[str, Any]:
    closed = 0
    for pool in _iter_provider_key_pools(app):
        close = getattr(pool, "close", None)
        if callable(close):
            await close()
            closed += 1
    snapshot = provider_key_pools_snapshot(app)
    snapshot["closed"] = closed
    return snapshot


def api_key_has_model_rules(app: FastAPI, api_index: int) -> bool:
    config = getattr(app.state, "config", {}) or {}
    runtime_config = getattr(app.state, "runtime_config", None)
    model_rules = getattr(runtime_config, "api_key_model_rules_by_index", None)
    if model_rules is not None and getattr(app.state, "runtime_config_source_id", None) == id(config):
        return 0 <= api_index < len(model_rules) and bool(model_rules[api_index])
    return bool(safe_get(config, "api_keys", api_index, "model"))


def get_runtime_api_list() -> list[str]:
    runtime_config = getattr(app.state, "runtime_config", None)
    config_api_list = getattr(runtime_config, "api_list", None)
    if config_api_list:
        return list(config_api_list)
    runtime_api_list = getattr(app.state, "api_list", None)
    if runtime_api_list:
        return runtime_api_list
    config = getattr(app.state, "config", {}) or {}
    return [item.get("api") for item in config.get("api_keys", []) if item.get("api")]


def get_runtime_api_key_index() -> dict[str, int]:
    api_key_index = getattr(app.state, "api_key_index", None)
    if api_key_index is not None:
        return api_key_index
    api_list = get_runtime_api_list()
    api_key_index = {api_key: index for index, api_key in enumerate(api_list)}
    app.state.api_key_index = api_key_index
    return api_key_index

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
    return await stats_repository.compute_total_cost(
        filter_api_key=filter_api_key,
        start_dt=start_dt_obj,
    )

async def update_paid_api_keys_states(app, paid_key):
    """
    更新付费API密钥的状态

    参数:
    app - FastAPI应用实例
    check_index - API密钥在配置中的索引
    paid_key - 需要更新状态的API密钥
    """
    api_key_index = getattr(app.state, "api_key_index", None)
    if api_key_index is None:
        api_key_index = {api_key: index for index, api_key in enumerate(getattr(app.state, "api_list", []) or [])}
        app.state.api_key_index = api_key_index
    check_index = api_key_index.get(paid_key)
    if check_index is None:
        raise HTTPException(
            status_code=403,
            detail={"error": "Invalid or missing API Key"}
        )
    credits = safe_get(app.state.config, 'api_keys', check_index, "preferences", "credits", default=-1)
    created_at = safe_get(app.state.config, 'api_keys', check_index, "preferences", "created_at", default=datetime.now(timezone.utc) - timedelta(days=30))
    created_at = created_at.astimezone(timezone.utc)

    state, total_cost = await compute_paid_api_key_state(
        credits=credits,
        created_at=created_at,
        paid_key=paid_key,
        compute_total_cost=compute_total_cost_from_db,
        get_usage_data=get_usage_data,
    )
    if state is not None:
        app.state.paid_api_keys_states[paid_key] = state.to_dict()

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
    provider_pool_snapshot = await close_provider_key_pools(app)
    stream_cleanup_snapshot = await wait_background_stream_cleanup_tasks(timeout=5.0)
    logger.info(
        "Shutdown cleanup status: provider_key_pools=%s stream_cleanup=%s",
        json.dumps(provider_pool_snapshot, ensure_ascii=False, default=str),
        json.dumps(stream_cleanup_snapshot, ensure_ascii=False, default=str),
    )
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

stats_repository = StatsRepository(
    async_session,
    semaphore=db_semaphore,
    debug=is_debug,
)

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
        await stats_repository.add_request_stat(current_info)
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
        await stats_repository.add_channel_stat(
            request_id=request_id,
            provider=provider,
            model=model,
            api_key=api_key,
            provider_api_key=provider_api_key,
            success=success,
        )
    except Exception as e:
        logger.error(f"Error acquiring database lock: {str(e)}")
        if is_debug:
            import traceback
            traceback.print_exc()

async def get_api_key(request: Request):
    return extract_api_key_from_headers(request.headers)

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


async def _moderate_content_for_middleware(
    request: ModerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int,
):
    return await moderations(request, background_tasks, api_index)


# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有头部字段
)

app.add_middleware(
    StatsMiddleware,
    dependencies=StatsMiddlewareDependencies(
        app_state=app.state,
        database_disabled=DISABLE_DATABASE,
        runtime_gauges=runtime_gauges,
        trace_factory=RequestTrace,
        incoming_trace_context=_incoming_trace_context,
        get_api_key=get_api_key,
        get_client_ip=get_client_ip,
        parse_request_body=parse_request_body,
        message_role_summary=_message_role_summary,
        messages_request_last_text=_messages_request_last_text,
        is_public_health_request=_is_public_health_request,
        is_video_or_asset_request_path=lambda path: _is_video_or_asset_request_path(path),
        lingjing_request_model_for_openapi=lambda payload, query_params=None: _lingjing_request_model_for_openapi(payload, query_params),
        video_prompt_from_body=lambda request_body: _video_prompt_from_body(request_body),
        monitor_disconnect=monitor_disconnect,
        log_debug_request_headers=_log_debug_request_headers,
        log_debug_request_body=_log_debug_request_body,
        mask_secret_for_log=lambda value: _mask_secret_for_log(value),
        update_stats=update_stats,
        emit_request_observability=_emit_request_observability,
        mark_first_byte_observed=_mark_first_byte_observed,
        moderation_handler=_moderate_content_for_middleware,
        logging_response_class=LoggingStreamingResponse,
        debug=is_debug,
    ),
)

app.add_middleware(RequestBodyDecompressionMiddleware)

@app.middleware("http")
async def ensure_config(request: Request, call_next):
    if _is_public_health_request(request):
        return await call_next(request)
    if not hasattr(app.state, "global_rate_limit"):
        app.state.global_rate_limit = parse_rate_limit(
            safe_get(getattr(app.state, "config", {}) or {}, "preferences", "rate_limit", default="999999/min")
        )
    if (
        app
        and getattr(app.state, "api_keys_db", None)
        and (
            not hasattr(app.state, "runtime_config")
            or getattr(app.state, "runtime_config_source_id", None) != id(getattr(app.state, "config", None))
        )
    ):
        await refresh_runtime_state(app)
    return await call_next(request)


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return await healthz_response(VERSION)


@app.get("/v1/observability/runtime", include_in_schema=False)
async def observability_runtime():
    return await observability_runtime_response(
        runtime_gauges,
        getattr(app.state, "client_manager", None),
        stream_cleanup_snapshot=background_stream_cleanup_snapshot,
        provider_key_pools_snapshot=lambda: provider_key_pools_snapshot(app),
    )


class ClientManager(ClientPool):
    def __init__(self, pool_size=100):
        super().__init__(
            pool_size=pool_size,
            sweep_client=_sweep_httpx_client_idle_connections,
            current_trace=_current_trace,
            begin_upstream_pool=runtime_gauges.begin_upstream_pool,
            end_upstream_pool=runtime_gauges.end_upstream_pool,
        )

rate_limiter = InMemoryRateLimiter()

async def rate_limit_dependency():
    global_rate_limit = getattr(app.state, "global_rate_limit", parse_rate_limit("999999/min"))
    if await rate_limiter.is_rate_limited("global", global_rate_limit):
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

_CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
_CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_CODEX_OAUTH_REFRESH_SKEW_SECONDS = 30
_CODEX_REFRESH_TOKEN_STORE_PATH = os.getenv("CODEX_REFRESH_TOKEN_STORE_PATH", "./data/codex_refresh_tokens.json")


def _codex_client_getter(url: str, proxy: Optional[str]):
    return app.state.client_manager.get_client(url, proxy)


_codex_refresh_token_store_obj = CodexRefreshTokenStore(
    _CODEX_REFRESH_TOKEN_STORE_PATH,
    logger=logger,
)
_codex_oauth_manager = CodexOAuthTokenManager(
    refresh_token_store=_codex_refresh_token_store_obj,
    client_getter=_codex_client_getter,
    token_url=_CODEX_OAUTH_TOKEN_URL,
    client_id=_CODEX_OAUTH_CLIENT_ID,
    refresh_skew_seconds=_CODEX_OAUTH_REFRESH_SKEW_SECONDS,
)

# Backward-compatible module globals for existing tests and call sites.
_codex_oauth_cache = _codex_oauth_manager.cache
_codex_oauth_locks = _codex_oauth_manager.locks
_codex_refresh_token_store = _codex_refresh_token_store_obj.tokens
_codex_refresh_token_store_lock = _codex_refresh_token_store_obj._lock


def _split_codex_api_key(raw_api_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    return split_codex_api_key(raw_api_key)

async def _ensure_codex_refresh_token_store_loaded() -> None:
    await _codex_refresh_token_store_obj.ensure_loaded()

async def _reload_codex_refresh_token_store() -> None:
    await _codex_refresh_token_store_obj.reload()

async def _get_codex_refresh_token_from_store(provider_api_key_raw: Optional[str], *, force_reload: bool = False) -> Optional[str]:
    return await _codex_refresh_token_store_obj.get(provider_api_key_raw, force_reload=force_reload)

async def _persist_codex_refresh_token(provider_api_key_raw: Optional[str], refresh_token: Optional[str]) -> None:
    await _codex_refresh_token_store_obj.persist(provider_api_key_raw, refresh_token)

def _codex_oauth_lock(key: str) -> asyncio.Lock:
    return _codex_oauth_manager._lock_for(key)

def _codex_access_token_is_valid(entry: dict[str, Any]) -> bool:
    return _codex_oauth_manager.access_token_is_valid(entry)

async def _refresh_codex_access_token(refresh_token: str, proxy: Optional[str]) -> dict[str, Any]:
    return await _codex_oauth_manager.refresh_access_token(refresh_token, proxy)

async def _get_codex_access_token(provider_name: str, provider_api_key_raw: str, proxy: Optional[str]) -> str:
    return await _codex_oauth_manager.get_access_token(provider_name, provider_api_key_raw, proxy)


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
    current_info: Optional[dict[str, Any]] = None,
    http_request: Optional[Request] = None,
):
    timeout_value = int(timeout_value)
    provider_registry = getattr(app.state, "provider_registry", None)
    if provider_registry is None:
        provider_registry = ProviderRegistry(default_provider_adapters())
        app.state.provider_registry = provider_registry
    prepared = await prepare_provider_request(
        request=request,
        provider=provider,
        endpoint=endpoint,
        provider_api_key_raw=provider_api_key_raw,
        runtime_api_list=get_runtime_api_list(),
        config=app.state.config,
        provider_registry=provider_registry,
        select_provider_api_key_raw=select_provider_api_key_raw,
        resolve_codex_upstream_auth=_resolve_codex_upstream_auth,
        http_request=http_request,
    )
    original_model = prepared.original_model
    engine = prepared.engine
    channel_id = prepared.channel_id
    proxy = prepared.proxy
    provider_api_key_raw = prepared.provider_api_key_raw
    url = prepared.url
    headers = prepared.headers
    payload = prepared.payload
    last_message_role = prepared.last_message_role

    if not isinstance(current_info, dict):
        current_info = get_request_info()
    trace = _coerce_request_trace(current_info)
    if isinstance(current_info, dict):
        current_info["stream"] = bool(getattr(request, "stream", False))
        current_info["role"] = role
        _mark_current_info_stage(current_info, "provider_selected")
        _set_current_info_trace_tag(current_info, "provider", channel_id)
        _set_current_info_trace_tag(current_info, "model", request.model)
    if engine != "moderation":
        _log_stdout_request_summary(channel_id, request.model, engine, role)
    _add_trace_headers(headers, current_info)
    _mark_current_info_stage(current_info, "provider_key_selected")

    # print("proxy", proxy)

    try:
        async with app.state.client_manager.get_client(url, proxy, http2=False if engine == "codex" else None) as client:
            downstream_stream = bool(getattr(request, "stream", None))
            force_collect_codex_stream = engine == "codex" and not downstream_stream and endpoint is None
            upstream_response_headers: dict[str, str] = {}

            def capture_upstream_response_headers(headers: Any) -> None:
                upstream_response_headers.update(_copy_upstream_response_headers(headers))

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
                _mark_current_info_stage(current_info, "upstream_send_start")
                runtime_gauges.begin_waiting_first_byte(current_info)
                generator = fetch_response_stream(
                    client,
                    url,
                    headers,
                    payload,
                    engine,
                    original_model,
                    timeout_value,
                    response_headers_sink=capture_upstream_response_headers,
                )
                _mark_current_info_stage(current_info, "upstream_headers_received")
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, True, app.state.error_triggers, keepalive_interval=keepalive_interval, last_message_role=last_message_role)
                if first_response_time == 3.1415:
                    wrapped_generator = _mark_first_byte_on_stream(wrapped_generator, current_info, skip_keepalive=True)
                else:
                    _mark_first_byte_observed(current_info)
                response = StarletteStreamingResponse(wrapped_generator, media_type="text/event-stream", headers=upstream_response_headers)
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
                _mark_current_info_stage(current_info, "upstream_send_start")
                runtime_gauges.begin_waiting_first_byte(current_info)
                generator = fetch_response_stream(
                    client,
                    url,
                    headers,
                    payload,
                    engine,
                    original_model,
                    timeout_value,
                    response_headers_sink=capture_upstream_response_headers,
                )
                _mark_current_info_stage(current_info, "upstream_headers_received")
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, True, app.state.error_triggers, keepalive_interval=keepalive_interval, last_message_role=last_message_role)
                if first_response_time != 3.1415:
                    _mark_first_byte_observed(current_info)
                json_data = await collect_openai_chat_completion_from_streaming_sse(wrapped_generator, model=original_model)
                _mark_first_byte_observed(current_info)
                response = StarletteStreamingResponse(iter([json_data]), media_type="application/json", headers=upstream_response_headers)
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
                _mark_current_info_stage(current_info, "upstream_send_start")
                runtime_gauges.begin_waiting_first_byte(current_info)
                generator = fetch_response(
                    client,
                    url,
                    headers,
                    payload,
                    engine,
                    original_model,
                    timeout_value,
                    response_headers_sink=capture_upstream_response_headers,
                )
                _mark_current_info_stage(current_info, "upstream_headers_received")
                wrapped_generator, first_response_time = await error_handling_wrapper(generator, channel_id, engine, False, app.state.error_triggers, keepalive_interval=keepalive_interval, last_message_role=last_message_role)
                _mark_first_byte_observed(current_info)

                # 处理音频和其他二进制响应
                if endpoint == "/v1/audio/speech":
                    if isinstance(wrapped_generator, bytes):
                        response = Response(content=wrapped_generator, media_type="audio/mpeg", headers=upstream_response_headers)
                else:
                    async with aclosing(wrapped_generator):
                        first_element = await anext(wrapped_generator)
                    _mark_first_byte_observed(current_info)
                    first_element = first_element.lstrip("data: ")
                    decoded_element = await asyncio.to_thread(json.loads, first_element)
                    encoded_element = await asyncio.to_thread(json.dumps, decoded_element)
                    response = StarletteStreamingResponse(iter([encoded_element]), media_type="application/json", headers=upstream_response_headers)

            # 更新成功计数和首次响应时间
            background_tasks.add_task(update_channel_stats, current_info["request_id"], channel_id, request.model, current_info["api_key"], success=True, provider_api_key=provider_api_key_raw)
            current_info["first_response_time"] = first_response_time
            current_info["success"] = True
            current_info["provider"] = channel_id
            setattr(response, "current_info", current_info)
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
        current_info: Optional[dict[str, Any]] = None,
    ):
        config = app.state.config
        request_model_name = request_data.model
        if not api_key_has_model_rules(app, api_index):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        if not isinstance(current_info, dict):
            current_info = get_request_info()
        _coerce_request_trace(current_info)
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
                engine_for_timeout, stream_mode_for_timeout = get_engine(
                    provider,
                    endpoint=endpoint,
                    original_model=original_model,
                )
                timeout_resolution = apply_timeout_policy(
                    base_timeout=int(local_timeout_value),
                    timeout_policy=getattr(app.state, "timeout_policy", {}),
                    provider_name=provider_name,
                    endpoint=routing_endpoint,
                    method="POST",
                    stream=bool(stream_mode_for_timeout) if stream_mode_for_timeout is not None else bool(getattr(request_data, "stream", False)),
                    engine=engine_for_timeout,
                    original_model=original_model,
                    request_model=request_model_name,
                    role=plan.role,
                )
                local_timeout_value = int(timeout_resolution["timeout_value"])
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
                    current_info=current_info,
                    http_request=http_request,
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
    return normalize_responses_upstream_url(base_url, engine)

def _normalize_responses_compact_upstream_url(base_url: str, engine: str) -> str:
    return normalize_responses_compact_upstream_url(base_url, engine)

def _normalize_messages_upstream_url(base_url: str) -> str:
    return normalize_messages_upstream_url(base_url)

VIDEO_TASKS_ENDPOINT = "/v1/video/tasks"
VIDEO_ASSETS_ENDPOINT = "/v1/assets"
VIDEO_ASSET_GROUPS_ENDPOINT = "/v1/asset-groups"
CONTENT_GENERATION_TASKS_ENDPOINT = VIDEO_TASKS_ENDPOINT
LINGJING_OPENAPI_ENDPOINT_PREFIX = "/v1/openapi"
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
    return normalize_content_generation_tasks_upstream_url(base_url, task_id)

def _is_lingjing_provider(provider: dict) -> bool:
    if str(provider.get("engine") or "").strip().lower() == "lingjing":
        return True
    parsed = urlparse(str(provider.get("base_url") or ""))
    return parsed.netloc.endswith("lingjingai.cn")

def _normalize_lingjing_openapi_upstream_url(base_url: str, openapi_path: str, query: str = "") -> str:
    return normalize_lingjing_openapi_upstream_url(base_url, openapi_path, query)

def _lingjing_upstream_query(raw_query: str) -> str:
    return lingjing_upstream_query(raw_query)

def _normalize_lingjing_draw_task_upstream_url(base_url: str, *, method: str, task_id: Optional[str] = None) -> str:
    return normalize_lingjing_draw_task_upstream_url(base_url, method=method, task_id=task_id)


def _provider_registry() -> ProviderRegistry:
    provider_registry = getattr(app.state, "provider_registry", None)
    if provider_registry is None:
        provider_registry = ProviderRegistry(default_provider_adapters())
        app.state.provider_registry = provider_registry
    return provider_registry


def _video_adapter_for(provider: dict[str, Any], provider_name: str):
    registry_adapter = _provider_registry().for_engine("content-generation")
    return registry_adapter.get_video_adapter(app.state.config, provider, provider_name)


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


def _optional_positive_timeout(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _httpx_timeout_from_policy(
    timeout_resolution: dict[str, Any],
    *,
    stream: bool,
    default_connect: float = 15.0,
    default_write: float = 30.0,
) -> Optional[httpx.Timeout]:
    policy = dict(timeout_resolution.get("timeout_policy") or {})
    if stream and not any(policy.get(key) is not None for key in ("connect", "write", "pool", "idle")):
        return None
    connect_timeout = _optional_positive_timeout(policy.get("connect")) or default_connect
    write_timeout = _optional_positive_timeout(policy.get("write")) or default_write
    pool_timeout = _optional_positive_timeout(policy.get("pool"))
    if stream:
        read_timeout = _optional_positive_timeout(policy.get("idle"))
    else:
        read_timeout = (
            _optional_positive_timeout(policy.get("idle"))
            or _optional_positive_timeout(policy.get("total"))
            or _optional_positive_timeout(timeout_resolution.get("timeout_value"))
        )
    return httpx.Timeout(
        timeout=None,
        connect=connect_timeout,
        read=read_timeout,
        write=write_timeout,
        pool=pool_timeout,
    )


async def _await_first_byte_deadline(
    awaitable: Awaitable[Any],
    *,
    timeout_seconds: Any = None,
    deadline: Optional[float] = None,
    total_timeout_seconds: Any = None,
    total_deadline: Optional[float] = None,
    satisfied: Optional[Callable[[], bool]] = None,
) -> Any:
    timeout = _optional_positive_timeout(timeout_seconds)
    total_timeout = _optional_positive_timeout(total_timeout_seconds)
    if deadline is None:
        if timeout is None:
            if total_deadline is None:
                return await awaitable
        else:
            deadline = asyncio.get_running_loop().time() + timeout
    if deadline is None and total_deadline is None:
        return await awaitable
    task = asyncio.create_task(awaitable)
    loop = asyncio.get_running_loop()
    first_byte_timeout_for_message = timeout if timeout is not None else (
        max(0.0, deadline - loop.time()) if deadline is not None else None
    )
    total_timeout_for_message = total_timeout if total_timeout is not None else (
        max(0.0, total_deadline - loop.time()) if total_deadline is not None else None
    )
    try:
        while True:
            first_byte_satisfied = satisfied is not None and satisfied()
            active_deadlines: list[tuple[str, float, Optional[float]]] = []
            if not first_byte_satisfied and deadline is not None:
                active_deadlines.append(("first byte", deadline, first_byte_timeout_for_message))
            if total_deadline is not None:
                active_deadlines.append(("total response", total_deadline, total_timeout_for_message))
            if not active_deadlines:
                return await task
            timeout_label, active_deadline, timeout_for_message = min(active_deadlines, key=lambda item: item[1])
            remaining = active_deadline - loop.time()
            if remaining <= 0:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                raise asyncio.TimeoutError(timeout_label)
            done, _ = await asyncio.wait({task}, timeout=min(0.05, remaining))
            if task in done:
                return task.result()
    except asyncio.TimeoutError as exc:
        timeout_label = exc.args[0] if exc.args else "first byte"
        timeout_for_message = (
            total_timeout_for_message if timeout_label == "total response" else first_byte_timeout_for_message
        )
        timeout_for_message = timeout_for_message if timeout_for_message is not None else 0
        raise httpx.ReadTimeout(
            f"Request timed out waiting for {timeout_label} after {timeout_for_message:g} seconds",
            request=httpx.Request("POST", "https://uni-api.local/upstream-timeout"),
        ) from exc


async def _await_stream_next_with_total_deadline(
    upstream_iter: Any,
    *,
    total_deadline: Optional[float],
    total_timeout_seconds: Any,
) -> bytes:
    if total_deadline is None:
        return await upstream_iter.__anext__()
    remaining = total_deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise httpx.ReadTimeout(
            f"Request timed out waiting for total response after {float(total_timeout_seconds):g} seconds",
            request=httpx.Request("POST", "https://uni-api.local/upstream-timeout"),
        )
    try:
        return await asyncio.wait_for(upstream_iter.__anext__(), timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise httpx.ReadTimeout(
            f"Request timed out waiting for total response after {float(total_timeout_seconds):g} seconds",
            request=httpx.Request("POST", "https://uni-api.local/upstream-timeout"),
        ) from exc

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


def _is_oaix_precommit_keepalive(chunk: bytes) -> bool:
    try:
        event_type, payload = _extract_responses_stream_event(chunk.decode("utf-8", errors="replace").strip())
    except Exception:
        return False
    if event_type != "keepalive" or not isinstance(payload, dict):
        return False
    if set(payload.keys()) != {"type", "sequence_number"}:
        return False
    return payload.get("type") == "keepalive" and payload.get("sequence_number") == 0


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
        execution = await ResponsesRequestExecution.create(
            handler=self,
            http_request=http_request,
            request_data=request_data,
            api_index=api_index,
            background_tasks=background_tasks,
            endpoint=endpoint,
        )
        return await execution.run()


@dataclass(slots=True)
class ResponsesRequestExecution:
    handler: Any
    http_request: Request
    request_data: ResponsesRequest
    api_index: int
    background_tasks: BackgroundTasks
    endpoint: str
    config: dict[str, Any]
    current_info: dict[str, Any]
    disconnect_event: Optional[asyncio.Event]
    request_id: str
    request_model_name: str
    plan: RoutingPlan
    runner: UpstreamRunner
    last_error_response: dict[str, Any] = field(default_factory=dict)
    stream_output_queue: Optional[asyncio.Queue[Any]] = None
    stream_response_headers: dict[str, str] = field(default_factory=dict)
    stream_done_sentinel: object = field(default_factory=object)
    stream_body_started: bool = False
    stream_keepalive_sent: bool = False
    stream_precommit_chunks: list[bytes] = field(default_factory=list)
    stream_stats_tasks: list[asyncio.Task[Any]] = field(default_factory=list)

    @classmethod
    async def create(
        cls,
        *,
        handler: Any,
        http_request: Request,
        request_data: ResponsesRequest,
        api_index: int,
        background_tasks: BackgroundTasks,
        endpoint: str,
    ) -> "ResponsesRequestExecution":
        config = app.state.config
        request_model_name = request_data.model
        if not api_key_has_model_rules(app, api_index):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = _request_state_current_info(http_request) or get_request_info()
        _coerce_request_trace(current_info)
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_id = _responses_request_id(current_info)
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            handler.last_provider_indices,
            handler.locks,
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
        return cls(
            handler=handler,
            http_request=http_request,
            request_data=request_data,
            api_index=api_index,
            background_tasks=background_tasks,
            endpoint=endpoint,
            config=config,
            current_info=current_info,
            disconnect_event=disconnect_event,
            request_id=request_id,
            request_model_name=request_model_name,
            plan=plan,
            runner=runner,
        )

    async def run(self):
        if self.request_data.stream:
            response = await self._run_stream()
        else:
            response = await self._run_attempts()
        if isinstance(response, Response):
            setattr(response, "current_info", self.current_info)
        return response

    async def _run_attempts(self):
        return await self.runner.run(
            self._execute_attempt,
            prepare_attempt=self._prepare_attempt,
            before_next_attempt=self._before_next_attempt,
            after_failure=self._after_failure,
            build_error_response=self._build_error_response,
            build_final_response=self._build_final_response,
            allow_channel_exclusion=True,
            should_cool_down=self._should_cool_down,
            on_retry=_record_retry_observability,
            on_cooldown=_record_cooldown_observability,
        )

    async def _run_stream(self):
        self.stream_output_queue = asyncio.Queue()
        worker_task = asyncio.create_task(self._stream_worker())
        first_item = await self.stream_output_queue.get()
        if first_item is self.stream_done_sentinel:
            return Response(content="", status_code=204)
        if isinstance(first_item, Response):
            with suppress(asyncio.CancelledError):
                await worker_task
            return first_item
        return StarletteStreamingResponse(
            self._stream_body(worker_task, first_item),
            media_type="text/event-stream",
            headers=self.stream_response_headers,
        )

    async def _stream_worker(self) -> None:
        assert self.stream_output_queue is not None
        try:
            response = await self._run_attempts()
            if isinstance(response, Response):
                if response.status_code == 204:
                    return
                if hasattr(response, "body_iterator"):
                    self.stream_response_headers = dict(response.headers)
                    for chunk in self.stream_precommit_chunks:
                        await self._emit_stream_chunk(chunk)
                    self.stream_precommit_chunks.clear()
                    async with aclosing(response.body_iterator):
                        async for chunk in response.body_iterator:
                            await self._emit_stream_chunk(chunk)
                    return
                if not self.stream_body_started:
                    self.stream_precommit_chunks.clear()
                    await self.stream_output_queue.put(response)
                    return
                if response.status_code != 499:
                    await self._emit_stream_chunk(_stream_error_event_from_response(response))
                    await self._emit_stream_chunk(b"data: [DONE]\n\n")
            elif response is not None:
                await self._emit_stream_chunk(response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._handle_stream_worker_error(exc)
        finally:
            if self.stream_stats_tasks:
                await asyncio.gather(*self.stream_stats_tasks, return_exceptions=True)
            await self.stream_output_queue.put(self.stream_done_sentinel)

    async def _stream_body(self, worker_task: asyncio.Task[Any], first_item: Any):
        assert self.stream_output_queue is not None
        try:
            yield first_item
            while True:
                item = await self.stream_output_queue.get()
                if item is self.stream_done_sentinel:
                    break
                yield item
        finally:
            if self.disconnect_event is not None:
                self.disconnect_event.set()
            if not worker_task.done():
                worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task

    async def _handle_stream_worker_error(self, exc: Exception) -> None:
        assert self.stream_output_queue is not None
        trace_logger.error(
            "%s stream worker failed request_id=%s model=%s error_type=%s: %s",
            self.endpoint,
            self.request_id,
            self.request_model_name,
            type(exc).__name__,
            str(exc) or type(exc).__name__,
        )
        error_message = str(exc) or type(exc).__name__
        if not self.stream_body_started:
            self.stream_precommit_chunks.clear()
            await self.stream_output_queue.put(
                JSONResponse(status_code=500, content={"error": error_message})
            )
            return
        await self._emit_stream_chunk(_build_responses_stream_error_event(500, error_message))
        await self._emit_stream_chunk(b"data: [DONE]\n\n")

    async def _emit_stream_chunk(self, chunk: Any) -> None:
        if self.stream_output_queue is None:
            return
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        if not isinstance(chunk, (bytes, bytearray)):
            chunk = str(chunk).encode("utf-8")
        self.stream_body_started = True
        await self.stream_output_queue.put(bytes(chunk))

    async def _emit_precommit_keepalive(
        self,
        upstream_keepalive: Optional[bytes],
        *,
        passthrough: bool = False,
    ) -> bool:
        if self.stream_output_queue is None:
            return False
        if self.stream_keepalive_sent:
            return True
        chunk = upstream_keepalive or _build_responses_stream_keepalive_event()
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        if not isinstance(chunk, (bytes, bytearray)):
            chunk = str(chunk).encode("utf-8")
        chunk_bytes = bytes(chunk)
        self.stream_keepalive_sent = True
        if passthrough and upstream_keepalive is not None and _is_oaix_precommit_keepalive(chunk_bytes):
            await self._emit_stream_chunk(chunk_bytes)
            _mark_first_byte_observed(self.current_info)
            return True
        self.stream_precommit_chunks.append(chunk_bytes)
        return True

    def _schedule_channel_stats(self, channel_id: str, *, success: bool, provider_api_key: Optional[str]) -> None:
        args = (
            self.current_info["request_id"],
            channel_id,
            self.request_model_name,
            self.current_info["api_key"],
        )
        kwargs = {"success": success, "provider_api_key": provider_api_key}
        if self.stream_output_queue is not None:
            self.stream_stats_tasks.append(asyncio.create_task(update_channel_stats(*args, **kwargs)))
        else:
            self.background_tasks.add_task(update_channel_stats, *args, **kwargs)

    async def _before_next_attempt(self):
        if self.stream_output_queue is not None and not self.stream_body_started:
            self.stream_precommit_chunks.clear()
            self.stream_keepalive_sent = False
        if self.disconnect_event is not None and self.disconnect_event.is_set():
            _log_responses_downstream_disconnect(
                self.endpoint,
                self.current_info,
                model_id=self.request_model_name,
                stage="before-provider-select",
            )
            return Response(content="", status_code=499)
        return None

    async def _prepare_attempt(self, attempt: Any) -> None:
        provider = attempt.provider
        provider_name = attempt.provider_name
        original_model = attempt.original_model
        engine, stream_mode = get_engine(provider, endpoint=self.endpoint, original_model=original_model)
        if stream_mode is not None:
            self.request_data.stream = stream_mode

        attempt.state["failure_stage"] = "validation"
        if engine not in ("gpt", "codex"):
            raise HTTPException(
                status_code=400,
                detail=f"{self.endpoint} only supports upstream engines: gpt/codex (got {engine})",
            )

        wants_compact = self.endpoint.rstrip("/").endswith("/compact")
        upstream_url = self._upstream_url(provider, engine, wants_compact)
        proxy = safe_get(self.config, "preferences", "proxy", default=None)
        proxy = safe_get(provider, "preferences", "proxy", default=proxy)
        channel_id = f"{provider_name}"
        _mark_current_info_stage(self.current_info, "provider_selected")
        _set_current_info_trace_tag(self.current_info, "provider", channel_id)
        _set_current_info_trace_tag(self.current_info, "model", self.request_model_name)

        commit_policy = safe_get(provider, "preferences", "responses_stream_commit_policy", default="real_output")
        attempt.state.update(
            {
                "upstream_url": upstream_url,
                "channel_id": channel_id,
                "engine": engine,
                "responses_stream_commit_policy": str(commit_policy or "real_output"),
                "failure_stage": "auth",
            }
        )
        attempt.provider_api_key_raw = await self.runner.select_provider_api_key(attempt)
        _mark_current_info_stage(self.current_info, "provider_key_selected")

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
            (original_model, self.request_model_name),
            DEFAULT_TIMEOUT,
        )
        timeout_resolution = apply_timeout_policy(
            base_timeout=int(timeout_value),
            timeout_policy=getattr(app.state, "timeout_policy", {}),
            provider_name=provider_name,
            endpoint=self.endpoint,
            method="POST",
            stream=bool(self.request_data.stream),
            engine=engine,
            original_model=original_model,
            request_model=self.request_model_name,
            role=self.plan.role,
        )
        attempt.state.update(
            {
                "proxy": proxy,
                "api_key": api_key,
                "codex_account_id": codex_account_id,
                "wants_compact": wants_compact,
                "timeout_value": int(timeout_resolution["timeout_value"]),
                "upstream_timeout": _httpx_timeout_from_policy(
                    timeout_resolution,
                    stream=bool(self.request_data.stream),
                ),
                "first_byte_timeout": int(timeout_resolution["first_byte_timeout"]),
                "idle_timeout": timeout_resolution["idle_timeout"],
                "total_timeout": timeout_resolution["total_timeout"],
                "timeout_policy": timeout_resolution["timeout_policy"],
                "timeout_policy_sources": timeout_resolution["timeout_policy_sources"],
                "timeout_adjusted_from": timeout_resolution["timeout_adjusted_from"],
            }
        )

    def _upstream_url(self, provider: dict[str, Any], engine: str, wants_compact: bool) -> str:
        if wants_compact:
            upstream_url = _normalize_responses_compact_upstream_url(provider.get("base_url", ""), engine)
        else:
            upstream_url = _normalize_responses_upstream_url(provider.get("base_url", ""), engine)
        if engine == "gpt" and "v1/responses" not in upstream_url:
            raise HTTPException(
                status_code=400,
                detail=f"{self.endpoint} requires provider base_url ending with /v1/responses (got {upstream_url})",
            )
        if wants_compact and "compact" not in upstream_url:
            raise HTTPException(
                status_code=400,
                detail=f"{self.endpoint} requires provider base_url ending with /v1/responses/compact (got {upstream_url})",
            )
        return upstream_url

    async def _execute_attempt(self, attempt: Any):
        provider = attempt.provider
        engine = attempt.state["engine"]
        upstream_url = attempt.state["upstream_url"]
        proxy = attempt.state["proxy"]
        headers = self._build_headers(attempt)
        payload = self._build_payload(attempt)
        json_payload = await asyncio.to_thread(json.dumps, payload)
        attempt.state["payload_bytes"] = len(json_payload.encode("utf-8"))
        self._record_upstream_attempt_start(attempt)
        self._log_attempt(attempt, headers, payload)

        async with app.state.client_manager.get_client(upstream_url, proxy, http2=False if engine == "codex" else None) as client:
            if self.request_data.stream:
                return await self._execute_stream_attempt(client, attempt, headers, json_payload)
            return await self._execute_non_stream_attempt(client, attempt, headers, json_payload)

    def _build_headers(self, attempt: Any) -> dict[str, str]:
        engine = attempt.state["engine"]
        api_key = attempt.state["api_key"]
        codex_account_id = attempt.state["codex_account_id"]
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if engine == "codex":
            headers.setdefault("Openai-Beta", self.http_request.headers.get("Openai-Beta") or "responses=experimental")
            headers.setdefault("Originator", self.http_request.headers.get("Originator") or "codex_cli_rs")
            headers.setdefault("Version", CODEX_CLI_VERSION)
            headers.setdefault("Session_id", self.http_request.headers.get("Session_id") or str(uuid.uuid4()))
            headers.setdefault("User-Agent", CODEX_USER_AGENT)
            headers.setdefault("Accept", "text/event-stream" if self.request_data.stream else "application/json")
            if codex_account_id:
                headers.setdefault("Chatgpt-Account-Id", str(codex_account_id))
        apply_provider_preference_headers(headers, attempt.provider, http_request=self.http_request)
        if engine == "codex":
            force_codex_client_headers(headers)
        _add_trace_headers(headers, self.current_info)
        return headers

    def _build_payload(self, attempt: Any) -> dict[str, Any]:
        engine = attempt.state["engine"]
        payload = self.request_data.model_dump(exclude_unset=True)
        payload["model"] = attempt.original_model
        if engine == "codex":
            payload.pop("previous_response_id", None)
            payload.pop("prompt_cache_retention", None)
            payload.pop("safety_identifier", None)
            payload.setdefault("instructions", "")
        apply_post_body_parameter_overrides(
            payload,
            attempt.provider,
            self.request_model_name,
            skip_keys={"translation_options"},
        )
        if engine == "codex":
            strip_unsupported_codex_payload_fields(payload, strip_store=attempt.state["wants_compact"])
        return payload

    def _record_upstream_attempt_start(self, attempt: Any) -> None:
        attempts = self.current_info.get("upstream_attempts")
        if not isinstance(attempts, list):
            attempts = []
            self.current_info["upstream_attempts"] = attempts
        if len(attempts) >= 16:
            attempt.state["observability_attempt_index"] = None
            return

        trace = _coerce_request_trace(self.current_info)
        started_ms = None
        if isinstance(trace, RequestTrace):
            started_ms = max(0, int((time() - trace.started_at) * 1000))
            trace.add_ms("upstream_payload_bytes", attempt.state.get("payload_bytes", 0))
            trace.add_ms("upstream_timeout_seconds", attempt.state.get("timeout_value", 0))
            if attempt.state.get("timeout_adjusted_from") is not None:
                trace.add_ms("upstream_timeout_adjusted_from_seconds", attempt.state["timeout_adjusted_from"])
            merge_timing_spans(self.current_info, trace.snapshot())
        elif isinstance(self.current_info.get("start_time"), (int, float)):
            started_ms = max(0, int((time() - float(self.current_info["start_time"])) * 1000))

        upstream_host = urlparse(str(attempt.state.get("upstream_url") or "")).netloc
        entry = {
            "index": len(attempts) + 1,
            "endpoint": self.endpoint,
            "provider": attempt.state.get("channel_id", attempt.provider_name),
            "model": self.request_model_name,
            "actual_model": attempt.original_model,
            "engine": attempt.state.get("engine"),
            "upstream_host": upstream_host,
            "payload_bytes": int(attempt.state.get("payload_bytes") or 0),
            "timeout_seconds": int(attempt.state.get("timeout_value") or 0),
            "wants_compact": bool(attempt.state.get("wants_compact")),
            "stream": bool(self.request_data.stream),
        }
        if started_ms is not None:
            entry["started_ms"] = started_ms
        if attempt.state.get("timeout_adjusted_from") is not None:
            entry["timeout_adjusted_from_seconds"] = int(attempt.state["timeout_adjusted_from"])
        if attempt.state.get("timeout_policy_sources"):
            entry["timeout_policy_sources"] = list(attempt.state["timeout_policy_sources"])
        attempts.append(entry)
        attempt.state["observability_attempt_index"] = len(attempts) - 1

    def _record_upstream_attempt_result(
        self,
        attempt: Any,
        *,
        status_code: int,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        attempts = self.current_info.get("upstream_attempts")
        index = attempt.state.get("observability_attempt_index")
        if not isinstance(attempts, list) or not isinstance(index, int) or index < 0 or index >= len(attempts):
            return
        entry = attempts[index]
        if not isinstance(entry, dict):
            return
        entry["status_code"] = int(status_code)
        entry["success"] = bool(success)
        if error_type:
            entry["error_type"] = str(error_type)[:80]
        trace = _coerce_request_trace(self.current_info)
        if isinstance(trace, RequestTrace) and isinstance(entry.get("started_ms"), int):
            entry["duration_ms"] = max(0, int((time() - trace.started_at) * 1000) - int(entry["started_ms"]))
        elif isinstance(self.current_info.get("start_time"), (int, float)) and isinstance(entry.get("started_ms"), int):
            entry["duration_ms"] = max(
                0,
                int((time() - float(self.current_info["start_time"])) * 1000) - int(entry["started_ms"]),
            )

    def _log_attempt(self, attempt: Any, headers: dict[str, str], payload: dict[str, Any]) -> None:
        channel_id = attempt.state["channel_id"]
        upstream_url = attempt.state["upstream_url"]
        engine = attempt.state["engine"]
        _log_stdout_request_summary(channel_id, self.request_model_name, engine, self.plan.role)
        trace_logger.info(
            "endpoint=%s request_id=%s provider=%-11s model=%-22s engine=%-13s role=%s timeout_seconds=%s payload_bytes=%s upstream_url=%s",
            self.endpoint,
            self.request_id,
            channel_id[:11],
            self.request_model_name,
            engine[:13],
            self.plan.role,
            attempt.state.get("timeout_value"),
            attempt.state.get("payload_bytes"),
            upstream_url,
        )
        attempt.state["failure_stage"] = "upstream"
        attempt.state["track_channel_stats"] = True
        _log_debug_request_headers(
            "DEBUG upstream request headers",
            headers,
            endpoint=self.endpoint,
            upstream_url=upstream_url,
            provider=channel_id,
            model=self.request_model_name,
            actual_model=attempt.original_model,
        )
        _log_debug_request_body(
            "DEBUG upstream request body",
            payload,
            endpoint=self.endpoint,
            upstream_url=upstream_url,
            provider=channel_id,
            model=self.request_model_name,
            actual_model=attempt.original_model,
        )

    async def _execute_stream_attempt(self, client: Any, attempt: Any, headers: dict[str, str], json_payload: str):
        _mark_current_info_stage(self.current_info, "upstream_send_start")
        runtime_gauges.begin_waiting_first_byte(self.current_info)
        first_byte_timeout = _optional_positive_timeout(attempt.state.get("first_byte_timeout"))
        first_byte_deadline = (
            asyncio.get_running_loop().time() + first_byte_timeout
            if first_byte_timeout is not None
            else None
        )
        total_timeout = _optional_positive_timeout(attempt.state.get("total_timeout"))
        total_deadline = (
            asyncio.get_running_loop().time() + total_timeout
            if total_timeout is not None
            else None
        )
        stream_kwargs = {
            "method": "POST",
            "url": attempt.state["upstream_url"],
            "headers": headers,
            "content": json_payload,
        }
        if attempt.state.get("upstream_timeout") is not None:
            stream_kwargs["timeout"] = attempt.state["upstream_timeout"]
        stream_cm = client.stream(**stream_kwargs)
        upstream_resp = await _await_first_byte_deadline(
            stream_cm.__aenter__(),
            timeout_seconds=first_byte_timeout,
            deadline=first_byte_deadline,
            total_timeout_seconds=total_timeout,
            total_deadline=total_deadline,
        )
        _mark_current_info_stage(self.current_info, "upstream_headers_received")
        if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            raw = await upstream_resp.aread()
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            raise HTTPException(status_code=upstream_resp.status_code, detail=raw.decode("utf-8", errors="replace"))

        if self.stream_output_queue is not None:
            self.stream_response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        upstream_iter = upstream_resp.aiter_bytes()
        try:
            precommit_keepalive_callback = None
            if self.stream_output_queue is not None:
                precommit_keepalive_callback = functools.partial(
                    self._emit_precommit_keepalive,
                    passthrough=attempt.state.get("engine") == "codex",
                )
            buffered_chunks, stream_committed = await _await_first_byte_deadline(
                _prime_responses_upstream_stream(
                    upstream_iter,
                    disconnect_event=self.disconnect_event,
                    commit_policy=attempt.state.get("responses_stream_commit_policy", "real_output"),
                    precommit_keepalive_callback=precommit_keepalive_callback,
                ),
                timeout_seconds=first_byte_timeout,
                deadline=first_byte_deadline,
                total_timeout_seconds=total_timeout,
                total_deadline=total_deadline,
                satisfied=lambda: self.stream_body_started,
            )
            _mark_first_byte_observed(self.current_info)
        except HTTPException:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            raise
        except RESPONSES_STREAM_NETWORK_ERRORS:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            raise
        except BaseException:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            raise

        if self.disconnect_event is not None and self.disconnect_event.is_set():
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            _log_responses_downstream_disconnect(
                self.endpoint,
                self.current_info,
                model_id=self.request_model_name,
                provider_name=attempt.provider_name,
                stage="before-stream-commit",
            )
            return Response(content="", status_code=499)

        self._record_upstream_attempt_result(attempt, status_code=upstream_resp.status_code, success=True)
        self._mark_success(attempt.state["channel_id"], attempt.provider_api_key_raw)
        response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        return StarletteStreamingResponse(
            self._proxy_responses_stream(
                attempt,
                buffered_chunks,
                upstream_iter,
                stream_cm,
                upstream_resp,
                stream_committed,
                total_deadline=total_deadline,
                total_timeout_seconds=total_timeout,
            ),
            media_type="text/event-stream",
            headers=response_headers,
        )

    async def _proxy_responses_stream(
        self,
        attempt: Any,
        buffered_chunks: list[bytes],
        upstream_iter: Any,
        stream_cm: Any,
        upstream_resp: Any,
        stream_committed: bool,
        *,
        total_deadline: Optional[float] = None,
        total_timeout_seconds: Any = None,
    ):
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
                _mark_first_byte_observed(self.current_info)
                if self._downstream_disconnected(attempt, stage="after-stream-commit"):
                    return
                track_responses_events(chunk)
                yield chunk
            while True:
                try:
                    chunk = await _await_stream_next_with_total_deadline(
                        upstream_iter,
                        total_deadline=total_deadline,
                        total_timeout_seconds=total_timeout_seconds,
                    )
                except StopAsyncIteration:
                    break
                _mark_first_byte_observed(self.current_info)
                if self._downstream_disconnected(attempt, stage="after-stream-commit"):
                    break
                track_responses_events(chunk)
                yield chunk
        except RESPONSES_STREAM_NETWORK_ERRORS as exc:
            await self._handle_proxy_stream_abort(attempt, exc, stream_committed)
            if stream_committed:
                yield b"data: [DONE]\n\n"
        finally:
            if not completed_seen or not usage_seen:
                trace_logger.warning(
                    "%s upstream stream finished without completed usage request_id=%s model=%s provider=%s output_seen=%s completed_seen=%s usage_seen=%s upstream_url=%s",
                    self.endpoint,
                    self.request_id,
                    self.request_model_name,
                    attempt.provider_name,
                    output_seen,
                    completed_seen,
                    usage_seen,
                    attempt.state["upstream_url"],
                )
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)

    def _downstream_disconnected(self, attempt: Any, *, stage: str) -> bool:
        if self.disconnect_event is None or not self.disconnect_event.is_set():
            return False
        _log_responses_downstream_disconnect(
            self.endpoint,
            self.current_info,
            model_id=self.request_model_name,
            provider_name=attempt.provider_name,
            stage=stage,
        )
        return True

    async def _handle_proxy_stream_abort(self, attempt: Any, exc: Exception, stream_committed: bool) -> None:
        stream_stage = "post-commit" if stream_committed else "preflight"
        error_text = str(exc) or type(exc).__name__
        request_model, actual_model = _log_model_names(self.request_model_name, attempt.original_model)
        trace_logger.warning(
            "%s upstream stream aborted stage=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
            self.endpoint,
            stream_stage,
            type(exc).__name__,
            self.request_id,
            request_model,
            actual_model,
            attempt.provider_name,
            _mask_secret_for_log(attempt.provider_api_key_raw),
            attempt.state["upstream_url"],
            error_text,
        )

    async def _execute_non_stream_attempt(self, client: Any, attempt: Any, headers: dict[str, str], json_payload: str):
        _mark_current_info_stage(self.current_info, "upstream_send_start")
        runtime_gauges.begin_waiting_first_byte(self.current_info)
        try:
            upstream_resp = await client.post(
                attempt.state["upstream_url"],
                headers=headers,
                content=json_payload,
                timeout=attempt.state["timeout_value"],
            )
            _mark_current_info_stage(self.current_info, "upstream_headers_received")
            _mark_first_byte_observed(self.current_info)
        except Exception:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            raise
        if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
            raw = await upstream_resp.aread()
            raise HTTPException(status_code=upstream_resp.status_code, detail=raw.decode("utf-8", errors="replace"))

        data = upstream_resp.json()
        semantic_failure = _responses_failure_http_exception(data)
        if semantic_failure is not None:
            raise semantic_failure

        self._record_upstream_attempt_result(attempt, status_code=upstream_resp.status_code, success=True)
        self._mark_success(attempt.state["channel_id"], attempt.provider_api_key_raw)
        response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        return JSONResponse(status_code=upstream_resp.status_code, content=data, headers=response_headers)

    def _mark_success(self, channel_id: str, provider_api_key: Optional[str]) -> None:
        self._schedule_channel_stats(channel_id, success=True, provider_api_key=provider_api_key)
        self.current_info["first_response_time"] = 0
        self.current_info["success"] = True
        self.current_info["provider"] = channel_id

    def _after_failure(self, attempt: Any, exc: Exception, status_code: int, error_message: Any) -> None:
        self._record_upstream_attempt_result(
            attempt,
            status_code=status_code,
            success=False,
            error_type=type(exc).__name__,
        )
        if attempt.state.get("track_channel_stats"):
            self._schedule_channel_stats(
                attempt.state["channel_id"],
                success=False,
                provider_api_key=attempt.provider_api_key_raw,
            )
        upstream_url = attempt.state.get("upstream_url", "")
        failure_stage = attempt.state.get("failure_stage")
        request_model, actual_model = _log_model_names(self.request_model_name, attempt.original_model)
        if failure_stage == "auth" and isinstance(exc, ValueError):
            trace_logger.error(
                "%s invalid codex api key request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
                self.endpoint,
                self.request_id,
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
                self.endpoint,
                self.request_id,
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
            self.endpoint,
            status_code,
            type(exc).__name__,
            self.request_id,
            request_model,
            actual_model,
            attempt.state.get("channel_id", attempt.provider_name),
            _mask_secret_for_log(attempt.provider_api_key_raw),
            upstream_url,
            error_message,
        )

    def _should_cool_down(self, exc: Exception, status_code: int, error_message: Any, attempt: Any) -> bool:
        _ = error_message, attempt
        return not isinstance(exc, ValueError) and status_code not in (400, 413)

    def _build_error_response(self, status_code: int, error_message: Any):
        self.current_info["first_response_time"] = -1
        self.current_info["success"] = False
        self.current_info["provider"] = None
        return build_upstream_error_response(
            status_code=status_code,
            error_message=error_message,
            fallback_prefix="Error: Current provider response failed",
        )

    def _build_final_response(self, completed_plan: Any):
        self.current_info["first_response_time"] = -1
        self.current_info["success"] = False
        self.current_info["provider"] = None
        return JSONResponse(
            status_code=completed_plan.status_code,
            content={"error": f"All {self.request_model_name} error: {completed_plan.error_message}"},
        )


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
        request_model_name = str(request_body.get("model") or "").strip()
        if not request_model_name:
            raise HTTPException(status_code=422, detail="Request body requires a model")

        config = app.state.config
        if not api_key_has_model_rules(app, api_index):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = get_request_info()
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
        ctx = {
            "http_request": http_request,
            "request_body": request_body,
            "request_model_name": request_model_name,
            "endpoint": endpoint,
            "config": config,
            "current_info": current_info,
            "disconnect_event": disconnect_event,
            "request_id": request_id,
            "plan": plan,
            "runner": runner,
            "background_tasks": background_tasks,
            "last_error_response": {},
        }

        return await runner.run(
            lambda attempt: self._messages_execute_attempt(attempt, ctx),
            prepare_attempt=lambda attempt: self._messages_prepare_attempt(attempt, ctx),
            before_next_attempt=lambda: self._messages_before_next_attempt(ctx),
            after_failure=lambda attempt, exc, status_code, error_message: self._messages_after_failure(
                attempt,
                exc,
                status_code,
                error_message,
                ctx,
            ),
            build_error_response=lambda status_code, error_message: self._messages_build_error_response(
                status_code,
                error_message,
                ctx,
            ),
            build_final_response=lambda completed_plan: self._messages_build_final_response(completed_plan, ctx),
            should_cool_down=self._messages_should_cool_down,
            on_retry=_record_retry_observability,
            on_cooldown=_record_cooldown_observability,
        )

    async def _messages_before_next_attempt(self, ctx: dict[str, Any]):
        disconnect_event = ctx["disconnect_event"]
        if disconnect_event is not None and disconnect_event.is_set():
            trace_logger.info(
                "%s downstream disconnect stage=before-provider-select request_id=%s model=%s",
                ctx["endpoint"],
                ctx["request_id"],
                ctx["request_model_name"],
            )
            return Response(content="", status_code=499)
        return None

    async def _messages_prepare_attempt(self, attempt: Any, ctx: dict[str, Any]) -> None:
        provider = attempt.provider
        provider_name = attempt.provider_name
        original_model = attempt.original_model
        endpoint = ctx["endpoint"]
        request_model_name = ctx["request_model_name"]
        engine, stream_mode = get_engine(provider, endpoint=endpoint, original_model=original_model)
        attempt.state["failure_stage"] = "validation"

        upstream_url = _normalize_messages_upstream_url(provider.get("base_url", ""))
        if not upstream_url:
            raise HTTPException(status_code=400, detail=f"{endpoint} requires provider base_url")

        upstream_path = urlparse(upstream_url).path.rstrip("/")
        is_messages_upstream = upstream_path.endswith("/v1/messages") or upstream_path.endswith("/messages")
        if engine != "claude" and not is_messages_upstream:
            raise HTTPException(status_code=400, detail=f"{endpoint} only supports upstream engine: claude (got {engine})")

        proxy = safe_get(ctx["config"], "preferences", "proxy", default=None)
        proxy = safe_get(provider, "preferences", "proxy", default=proxy)
        attempt.state.update(
            {
                "upstream_url": upstream_url,
                "channel_id": f"{provider_name}",
                "engine": "claude",
                "proxy": proxy,
                "stream_mode": stream_mode,
                "failure_stage": "auth",
            }
        )
        attempt.provider_api_key_raw = await ctx["runner"].select_provider_api_key(attempt)
        timeout_value = get_preference(
            app.state.provider_timeouts,
            provider_name,
            (original_model, request_model_name),
            DEFAULT_TIMEOUT,
        )
        timeout_resolution = apply_timeout_policy(
            base_timeout=int(timeout_value),
            timeout_policy=getattr(app.state, "timeout_policy", {}),
            provider_name=provider_name,
            endpoint=endpoint,
            method="POST",
            stream=bool(stream_mode) if stream_mode is not None else bool((ctx["request_body"] or {}).get("stream")),
            engine="claude",
            original_model=original_model,
            request_model=request_model_name,
            role=ctx["plan"].role,
        )
        attempt.state["api_key"] = attempt.provider_api_key_raw
        attempt.state["timeout_value"] = int(timeout_resolution["timeout_value"])
        attempt.state["timeout_policy_sources"] = timeout_resolution["timeout_policy_sources"]

    async def _messages_execute_attempt(self, attempt: Any, ctx: dict[str, Any]):
        provider = attempt.provider
        original_model = attempt.original_model
        upstream_url = attempt.state["upstream_url"]
        proxy = attempt.state["proxy"]
        timeout_value = attempt.state["timeout_value"]
        channel_id = attempt.state["channel_id"]
        request_model_name = ctx["request_model_name"]

        payload = dict(ctx["request_body"])
        payload["model"] = original_model
        if attempt.state.get("stream_mode") is not None:
            payload["stream"] = bool(attempt.state["stream_mode"])
        apply_post_body_parameter_overrides(payload, provider, request_model_name)

        headers = self._messages_headers(ctx["http_request"], provider, attempt.state["api_key"])
        self._messages_log_attempt(ctx, attempt, payload, headers)
        json_payload = await asyncio.to_thread(json.dumps, payload)

        async with app.state.client_manager.get_client(upstream_url, proxy) as client:
            if payload.get("stream"):
                return await self._messages_stream_response(client, attempt, ctx, headers, json_payload)
            return await self._messages_non_stream_response(client, attempt, ctx, headers, json_payload)

    def _messages_headers(self, http_request: Request, provider: dict[str, Any], api_key: Any) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": http_request.headers.get("anthropic-version") or "2023-06-01",
        }
        anthropic_beta = http_request.headers.get("anthropic-beta")
        if anthropic_beta:
            headers["anthropic-beta"] = anthropic_beta
        if api_key:
            headers["x-api-key"] = str(api_key)
        apply_provider_preference_headers(headers, provider, http_request=http_request)
        return headers

    def _messages_log_attempt(self, ctx: dict[str, Any], attempt: Any, payload: dict[str, Any], headers: dict[str, str]) -> None:
        channel_id = attempt.state["channel_id"]
        upstream_url = attempt.state["upstream_url"]
        request_model_name = ctx["request_model_name"]
        _log_stdout_request_summary(channel_id, request_model_name, "claude", ctx["plan"].role)
        trace_logger.info(
            "endpoint=%s request_id=%s provider=%-11s model=%-22s engine=%-13s role=%s upstream_url=%s",
            ctx["endpoint"],
            ctx["request_id"],
            channel_id[:11],
            request_model_name,
            "claude",
            ctx["plan"].role,
            upstream_url,
        )
        attempt.state["failure_stage"] = "upstream"
        attempt.state["track_channel_stats"] = True
        _log_debug_request_headers(
            "DEBUG upstream request headers",
            headers,
            endpoint=ctx["endpoint"],
            upstream_url=upstream_url,
            provider=channel_id,
            model=request_model_name,
            actual_model=attempt.original_model,
        )
        _log_debug_request_body(
            "DEBUG upstream request body",
            payload,
            endpoint=ctx["endpoint"],
            upstream_url=upstream_url,
            provider=channel_id,
            model=request_model_name,
            actual_model=attempt.original_model,
        )

    async def _messages_stream_response(self, client: Any, attempt: Any, ctx: dict[str, Any], headers: dict[str, str], json_payload: str):
        upstream_url = attempt.state["upstream_url"]
        stream_cm = client.stream("POST", upstream_url, headers=headers, content=json_payload, timeout=attempt.state["timeout_value"])
        upstream_resp = await stream_cm.__aenter__()
        response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
            raw = await upstream_resp.aread()
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            self._messages_set_last_error(ctx, raw, response_headers)
            raise HTTPException(status_code=upstream_resp.status_code, detail=raw.decode("utf-8", errors="replace"))

        upstream_iter = upstream_resp.aiter_raw()
        try:
            buffered_chunks = await _prime_passthrough_upstream_stream(upstream_iter, disconnect_event=ctx["disconnect_event"])
        except BaseException:
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            raise

        if ctx["disconnect_event"] is not None and ctx["disconnect_event"].is_set():
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            trace_logger.info(
                "%s downstream disconnect stage=before-stream-commit request_id=%s model=%s provider=%s",
                ctx["endpoint"],
                ctx["request_id"],
                ctx["request_model_name"],
                attempt.provider_name,
            )
            return Response(content="", status_code=499)

        self._messages_record_success(ctx, attempt)
        return StarletteStreamingResponse(
            self._messages_proxy_stream(ctx, attempt, buffered_chunks, upstream_iter, stream_cm, upstream_resp),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            media_type=response_headers.get("content-type", "text/event-stream"),
        )

    async def _messages_proxy_stream(self, ctx: dict[str, Any], attempt: Any, buffered_chunks: list[bytes], upstream_iter: Any, stream_cm: Any, upstream_resp: Any):
        try:
            for chunk in buffered_chunks:
                if self._messages_downstream_disconnected(ctx, attempt, stage="after-stream-commit"):
                    return
                yield chunk
            async for chunk in upstream_iter:
                if self._messages_downstream_disconnected(ctx, attempt, stage="after-stream-commit"):
                    break
                yield chunk
        finally:
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)

    def _messages_downstream_disconnected(self, ctx: dict[str, Any], attempt: Any, *, stage: str) -> bool:
        disconnect_event = ctx["disconnect_event"]
        if disconnect_event is None or not disconnect_event.is_set():
            return False
        trace_logger.info(
            "%s downstream disconnect stage=%s request_id=%s model=%s provider=%s",
            ctx["endpoint"],
            stage,
            ctx["request_id"],
            ctx["request_model_name"],
            attempt.provider_name,
        )
        return True

    async def _messages_non_stream_response(self, client: Any, attempt: Any, ctx: dict[str, Any], headers: dict[str, str], json_payload: str):
        upstream_resp = await client.post(
            attempt.state["upstream_url"],
            headers=headers,
            content=json_payload,
            timeout=attempt.state["timeout_value"],
        )
        response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        raw = upstream_resp.content
        if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
            self._messages_set_last_error(ctx, raw, response_headers)
            raise HTTPException(status_code=upstream_resp.status_code, detail=raw.decode("utf-8", errors="replace"))

        self._messages_record_success(ctx, attempt)
        return Response(
            content=raw,
            status_code=upstream_resp.status_code,
            headers=response_headers,
            media_type=response_headers.get("content-type", "application/json"),
        )

    def _messages_set_last_error(self, ctx: dict[str, Any], body: bytes, headers: dict[str, str]) -> None:
        ctx["last_error_response"].clear()
        ctx["last_error_response"].update({"body": body, "headers": headers})

    def _messages_record_success(self, ctx: dict[str, Any], attempt: Any) -> None:
        current_info = ctx["current_info"]
        channel_id = attempt.state["channel_id"]
        ctx["background_tasks"].add_task(
            update_channel_stats,
            current_info["request_id"],
            channel_id,
            ctx["request_model_name"],
            current_info["api_key"],
            success=True,
            provider_api_key=attempt.provider_api_key_raw,
        )
        current_info["first_response_time"] = 0
        current_info["success"] = True
        current_info["provider"] = channel_id

    def _messages_after_failure(self, attempt: Any, exc: Exception, status_code: int, error_message: Any, ctx: dict[str, Any]) -> None:
        current_info = ctx["current_info"]
        if attempt.state.get("track_channel_stats"):
            ctx["background_tasks"].add_task(
                update_channel_stats,
                current_info["request_id"],
                attempt.state["channel_id"],
                ctx["request_model_name"],
                current_info["api_key"],
                success=False,
                provider_api_key=attempt.provider_api_key_raw,
            )
        request_model, actual_model = _log_model_names(ctx["request_model_name"], attempt.original_model)
        trace_logger.error(
            "%s upstream error status=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
            ctx["endpoint"],
            status_code,
            type(exc).__name__,
            ctx["request_id"],
            request_model,
            actual_model,
            attempt.state.get("channel_id", attempt.provider_name),
            _mask_secret_for_log(attempt.provider_api_key_raw),
            attempt.state.get("upstream_url", ""),
            error_message,
        )

    def _messages_should_cool_down(self, exc: Exception, status_code: int, error_message: Any, attempt: Any) -> bool:
        _ = exc, error_message, attempt
        return status_code not in (400, 413)

    def _messages_build_error_response(self, status_code: int, error_message: Any, ctx: dict[str, Any]):
        current_info = ctx["current_info"]
        current_info["first_response_time"] = -1
        current_info["success"] = False
        current_info["provider"] = None
        last_error_response = ctx["last_error_response"]
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

    def _messages_build_final_response(self, completed_plan: Any, ctx: dict[str, Any]):
        current_info = ctx["current_info"]
        current_info["first_response_time"] = -1
        current_info["success"] = False
        current_info["provider"] = None
        return JSONResponse(
            status_code=completed_plan.status_code,
            content={"error": f"All {ctx['request_model_name']} error: {completed_plan.error_message}"},
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
        timeout_resolution = apply_timeout_policy(
            base_timeout=int(timeout_value),
            timeout_policy=getattr(app.state, "timeout_policy", {}),
            provider_name=provider_name,
            endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
            method=method,
            stream=False,
            engine="content-generation",
            original_model=original_model,
            request_model=request_model_name,
        )
        adapter = _video_adapter_for(provider, provider_name)
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
                timeout_value=int(timeout_resolution["timeout_value"]),
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
        current_info = get_request_info()
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
        if not api_key_has_model_rules(app, api_index):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = get_request_info()
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
        ctx = {
            "config": config,
            "current_info": current_info,
            "disconnect_event": disconnect_event,
            "request_id": request_id,
            "request_model_name": request_model_name,
            "request_body": request_body,
            "background_tasks": background_tasks,
            "method": method,
            "task_id": task_id,
            "plan": plan,
            "runner": runner,
            "last_error_response": {},
        }

        return await runner.run(
            lambda attempt: self._video_execute_attempt(attempt, ctx),
            prepare_attempt=lambda attempt: self._video_prepare_attempt(attempt, ctx),
            before_next_attempt=lambda: self._video_before_next_attempt(ctx),
            after_failure=lambda attempt, exc, status_code, error_message: self._video_after_failure(
                attempt,
                exc,
                status_code,
                error_message,
                ctx,
            ),
            build_error_response=lambda status_code, error_message: self._video_build_error_response(
                status_code,
                error_message,
                ctx,
            ),
            build_final_response=lambda completed_plan: self._video_build_final_response(completed_plan, ctx),
            should_cool_down=self._video_should_cool_down,
            on_retry=_record_retry_observability,
            on_cooldown=_record_cooldown_observability,
        )

    async def _video_before_next_attempt(self, ctx: dict[str, Any]):
        disconnect_event = ctx["disconnect_event"]
        if disconnect_event is not None and disconnect_event.is_set():
            trace_logger.info(
                "%s downstream disconnect stage=before-provider-select request_id=%s model=%s",
                CONTENT_GENERATION_TASKS_ENDPOINT,
                ctx["request_id"],
                ctx["request_model_name"],
            )
            return Response(content="", status_code=499)
        return None

    async def _video_prepare_attempt(self, attempt: Any, ctx: dict[str, Any]) -> None:
        provider = attempt.provider
        provider_name = attempt.provider_name
        original_model = attempt.original_model
        request_model_name = ctx["request_model_name"]
        engine, _ = get_engine(provider, endpoint=CONTENT_GENERATION_TASKS_ENDPOINT, original_model=original_model)
        attempt.state["failure_stage"] = "validation"
        if engine != "content-generation":
            raise HTTPException(
                status_code=400,
                detail=f"{CONTENT_GENERATION_TASKS_ENDPOINT} only supports upstream engine: content-generation (got {engine})",
            )

        proxy = safe_get(ctx["config"], "preferences", "proxy", default=None)
        proxy = safe_get(provider, "preferences", "proxy", default=proxy)
        attempt.state["failure_stage"] = "auth"
        attempt.provider_api_key_raw = await ctx["runner"].select_provider_api_key(attempt)
        adapter = _video_adapter_for(provider, provider_name)
        try:
            upstream_request = adapter.build_request(
                method=ctx["method"],
                task_id=ctx["task_id"],
                request_body=ctx["request_body"],
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

        timeout_value = get_preference(
            app.state.provider_timeouts,
            provider_name,
            (original_model, request_model_name),
            DEFAULT_TIMEOUT,
        )
        timeout_resolution = apply_timeout_policy(
            base_timeout=int(timeout_value),
            timeout_policy=getattr(app.state, "timeout_policy", {}),
            provider_name=provider_name,
            endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
            method=ctx["method"],
            stream=False,
            engine=engine,
            original_model=original_model,
            request_model=request_model_name,
            role=ctx["plan"].role,
        )
        attempt.state.update(
            {
                "video_adapter": adapter,
                "upstream_request": upstream_request,
                "upstream_url": upstream_request.url,
                "channel_id": f"{provider_name}",
                "engine": engine,
                "proxy": proxy,
                "api_key": attempt.provider_api_key_raw,
                "timeout_value": int(timeout_resolution["timeout_value"]),
                "timeout_policy_sources": timeout_resolution["timeout_policy_sources"],
            }
        )

    async def _video_execute_attempt(self, attempt: Any, ctx: dict[str, Any]):
        upstream_request = attempt.state["upstream_request"]
        payload = upstream_request.payload
        channel_id = attempt.state["channel_id"]
        self._video_log_attempt(attempt, ctx, upstream_request.headers, payload)
        upstream_resp = await self._send_upstream(
            method=upstream_request.method,
            upstream_url=attempt.state["upstream_url"],
            headers=upstream_request.headers,
            payload=payload,
            proxy=attempt.state["proxy"],
            timeout_value=attempt.state["timeout_value"],
        )
        return self._video_response_from_upstream(attempt, ctx, upstream_resp)

    def _video_log_attempt(self, attempt: Any, ctx: dict[str, Any], headers: dict[str, str], payload: Optional[dict[str, Any]]) -> None:
        channel_id = attempt.state["channel_id"]
        request_model_name = ctx["request_model_name"]
        _log_stdout_request_summary(channel_id, request_model_name, "content-generation", ctx["plan"].role)
        trace_logger.info(
            "endpoint=%s method=%s request_id=%s provider=%-11s model=%-22s engine=%-13s role=%s upstream_url=%s",
            CONTENT_GENERATION_TASKS_ENDPOINT,
            ctx["method"],
            ctx["request_id"],
            channel_id[:11],
            request_model_name,
            "content-generation",
            ctx["plan"].role,
            attempt.state["upstream_url"],
        )
        attempt.state["failure_stage"] = "upstream"
        attempt.state["track_channel_stats"] = True
        _log_debug_request_headers(
            "DEBUG upstream request headers",
            headers,
            endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
            upstream_url=attempt.state["upstream_url"],
            provider=channel_id,
            model=request_model_name,
            actual_model=attempt.original_model,
        )
        if payload is not None:
            _log_debug_request_body(
                "DEBUG upstream request body",
                payload,
                endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
                upstream_url=attempt.state["upstream_url"],
                provider=channel_id,
                model=request_model_name,
                actual_model=attempt.original_model,
            )

    def _video_response_from_upstream(self, attempt: Any, ctx: dict[str, Any], upstream_resp: httpx.Response) -> Response:
        adapter = attempt.state["video_adapter"]
        raw = upstream_resp.content
        normalized = adapter.normalize_response(
            method=ctx["method"],
            raw=raw,
            task_id=ctx["task_id"],
            request_model_name=ctx["request_model_name"],
            provider_name=attempt.provider_name,
            estimated_usage=_estimated_video_usage_from_request(ctx["request_body"]),
        )
        raw = normalized.raw
        response_media_type = normalized.media_type if _maybe_json_object(raw) else None
        self._video_remember_task_if_needed(attempt, ctx, normalized.task_id)
        if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
            return self._video_error_or_retry(attempt, ctx, upstream_resp, raw, response_media_type)
        if ctx["method"] == "DELETE" and ctx["task_id"]:
            self.task_routes.pop(ctx["task_id"], None)
        self._mark_result(
            background_tasks=ctx["background_tasks"],
            current_info=ctx["current_info"],
            channel_id=attempt.state["channel_id"],
            request_model_name=ctx["request_model_name"],
            success=True,
            provider_api_key_raw=attempt.provider_api_key_raw,
        )
        return self._raw_response(upstream_resp, raw, media_type=response_media_type)

    def _video_remember_task_if_needed(self, attempt: Any, ctx: dict[str, Any], normalized_task_id: Optional[str]) -> None:
        if ctx["method"] != "POST" or not normalized_task_id:
            return
        self._remember_task_route(
            task_id=normalized_task_id,
            request_model_name=ctx["request_model_name"],
            original_model=attempt.original_model,
            provider=attempt.provider,
            provider_name=attempt.provider_name,
            provider_api_key_raw=attempt.provider_api_key_raw,
            client_api_key=ctx["current_info"].get("api_key"),
            estimated_usage=_estimated_video_usage_from_request(ctx["request_body"]),
        )

    def _video_error_or_retry(self, attempt: Any, ctx: dict[str, Any], upstream_resp: httpx.Response, raw: bytes, response_media_type: Optional[str]) -> Response:
        if self._is_non_retryable_client_error(upstream_resp.status_code):
            self._mark_result(
                background_tasks=ctx["background_tasks"],
                current_info=ctx["current_info"],
                channel_id=attempt.state["channel_id"],
                request_model_name=ctx["request_model_name"],
                success=False,
                provider_api_key_raw=attempt.provider_api_key_raw,
            )
            return self._raw_response(upstream_resp, raw, media_type=response_media_type)
        ctx["last_error_response"].clear()
        ctx["last_error_response"].update({"body": raw, "headers": _copy_upstream_response_headers(upstream_resp.headers)})
        raise HTTPException(status_code=upstream_resp.status_code, detail=raw.decode("utf-8", errors="replace"))

    def _video_after_failure(self, attempt: Any, exc: Exception, status_code: int, error_message: Any, ctx: dict[str, Any]) -> None:
        current_info = ctx["current_info"]
        if attempt.state.get("track_channel_stats"):
            ctx["background_tasks"].add_task(
                update_channel_stats,
                current_info["request_id"],
                attempt.state["channel_id"],
                ctx["request_model_name"],
                current_info["api_key"],
                success=False,
                provider_api_key=attempt.provider_api_key_raw,
            )
        request_model, actual_model = _log_model_names(ctx["request_model_name"], attempt.original_model)
        trace_logger.error(
            "%s upstream error status=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s: %s",
            CONTENT_GENERATION_TASKS_ENDPOINT,
            status_code,
            type(exc).__name__,
            ctx["request_id"],
            request_model,
            actual_model,
            attempt.state.get("channel_id", attempt.provider_name),
            _mask_secret_for_log(attempt.provider_api_key_raw),
            attempt.state.get("upstream_url", ""),
            error_message,
        )

    def _video_should_cool_down(self, exc: Exception, status_code: int, error_message: Any, attempt: Any) -> bool:
        _ = exc, error_message, attempt
        return status_code in (401, 403, 429) or status_code >= 500

    def _video_build_error_response(self, status_code: int, error_message: Any, ctx: dict[str, Any]):
        current_info = ctx["current_info"]
        current_info["first_response_time"] = -1
        current_info["success"] = False
        current_info["provider"] = None
        last_error_response = ctx["last_error_response"]
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

    def _video_build_final_response(self, completed_plan: Any, ctx: dict[str, Any]):
        current_info = ctx["current_info"]
        current_info["first_response_time"] = -1
        current_info["success"] = False
        current_info["provider"] = None
        return JSONResponse(
            status_code=completed_plan.status_code,
            content={"error": f"All {ctx['request_model_name']} error: {completed_plan.error_message}"},
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
        current_info = get_request_info()
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
        ctx = {
            "http_request": http_request,
            "payload": payload,
            "request_model_name": request_model_name,
            "method_upper": method_upper,
            "openapi_path": openapi_path,
            "endpoint": endpoint,
            "config": config,
            "current_info": current_info,
            "request_id": request_id,
            "plan": plan,
            "runner": runner,
            "background_tasks": background_tasks,
            "last_error_response": {},
        }

        return await runner.run(
            lambda attempt: self._lingjing_execute_attempt(attempt, ctx),
            prepare_attempt=lambda attempt: self._lingjing_prepare_attempt(attempt, ctx),
            after_failure=lambda attempt, exc, status_code, error_message: self._lingjing_after_failure(
                attempt,
                exc,
                status_code,
                error_message,
                ctx,
            ),
            build_error_response=lambda status_code, error_message: self._lingjing_build_error_response(
                status_code,
                error_message,
                ctx,
            ),
            build_final_response=lambda completed_plan: self._lingjing_build_final_response(completed_plan, ctx),
            should_cool_down=self._lingjing_should_cool_down,
            on_retry=_record_retry_observability,
            on_cooldown=_record_cooldown_observability,
        )

    async def _lingjing_prepare_attempt(self, attempt: Any, ctx: dict[str, Any]) -> None:
        provider = attempt.provider
        provider_name = attempt.provider_name
        original_model = attempt.original_model
        endpoint = ctx["endpoint"]
        request_model_name = ctx["request_model_name"]
        attempt.state["failure_stage"] = "validation"
        if not _is_lingjing_provider(provider):
            raise HTTPException(status_code=400, detail=f"{endpoint} only supports Lingjing providers")

        upstream_url = _normalize_lingjing_openapi_upstream_url(
            provider.get("base_url", ""),
            ctx["openapi_path"],
            query=_lingjing_upstream_query(ctx["http_request"].url.query),
        )
        if not upstream_url:
            raise HTTPException(status_code=400, detail=f"{endpoint} requires provider base_url")

        proxy = safe_get(ctx["config"], "preferences", "proxy", default=None)
        proxy = safe_get(provider, "preferences", "proxy", default=proxy)
        attempt.state.update({"upstream_url": upstream_url, "channel_id": f"{provider_name}", "proxy": proxy, "failure_stage": "auth"})
        attempt.provider_api_key_raw = await ctx["runner"].select_provider_api_key(attempt)
        timeout_value = get_preference(
            app.state.provider_timeouts,
            provider_name,
            (original_model, request_model_name),
            DEFAULT_TIMEOUT,
        )
        timeout_resolution = apply_timeout_policy(
            base_timeout=int(timeout_value),
            timeout_policy=getattr(app.state, "timeout_policy", {}),
            provider_name=provider_name,
            endpoint=endpoint,
            method=ctx["method_upper"],
            stream=False,
            engine="lingjing",
            original_model=original_model,
            request_model=request_model_name,
            role=ctx["plan"].role,
        )
        attempt.state["api_key"] = attempt.provider_api_key_raw
        attempt.state["timeout_value"] = int(timeout_resolution["timeout_value"])
        attempt.state["timeout_policy_sources"] = timeout_resolution["timeout_policy_sources"]

    async def _lingjing_execute_attempt(self, attempt: Any, ctx: dict[str, Any]) -> Response:
        headers = _lingjing_headers(
            attempt.provider,
            attempt.state["api_key"],
            include_content_type=ctx["method_upper"] in {"POST", "PUT"},
        )
        outbound_payload = self._lingjing_outbound_payload(attempt, ctx)
        self._lingjing_log_attempt(attempt, ctx, headers, outbound_payload)
        upstream_resp = await self._send_upstream(
            method=ctx["method_upper"],
            upstream_url=attempt.state["upstream_url"],
            headers=headers,
            payload=outbound_payload,
            proxy=attempt.state["proxy"],
            timeout_value=attempt.state["timeout_value"],
        )
        return self._lingjing_response_from_upstream(attempt, ctx, upstream_resp)

    def _lingjing_outbound_payload(self, attempt: Any, ctx: dict[str, Any]) -> Optional[dict[str, Any]]:
        payload = ctx["payload"]
        if ctx["method_upper"] == "POST" and str(ctx["openapi_path"] or "").strip("/") == "draw/task/submit" and isinstance(payload, dict):
            outbound_payload = dict(payload)
            model_code = str(outbound_payload.get("modelCode") or "").strip()
            if not model_code or model_code == ctx["request_model_name"]:
                outbound_payload["modelCode"] = attempt.original_model
            outbound_payload.pop("model", None)
            outbound_payload.pop("request_model", None)
            return outbound_payload
        return payload

    def _lingjing_log_attempt(self, attempt: Any, ctx: dict[str, Any], headers: dict[str, str], outbound_payload: Optional[dict[str, Any]]) -> None:
        channel_id = attempt.state["channel_id"]
        trace_logger.info(
            "endpoint=%s method=%s request_id=%s provider=%-11s model=%-22s engine=%-13s role=%s upstream_url=%s",
            ctx["endpoint"],
            ctx["method_upper"],
            ctx["request_id"],
            channel_id[:11],
            ctx["request_model_name"],
            "lingjing",
            ctx["plan"].role,
            attempt.state["upstream_url"],
        )
        attempt.state["failure_stage"] = "upstream"
        attempt.state["track_channel_stats"] = True
        _log_debug_request_headers(
            "DEBUG upstream request headers",
            headers,
            endpoint=ctx["endpoint"],
            upstream_url=attempt.state["upstream_url"],
            provider=channel_id,
            model=ctx["request_model_name"],
            actual_model=attempt.original_model,
        )
        if outbound_payload is not None:
            _log_debug_request_body(
                "DEBUG upstream request body",
                outbound_payload,
                endpoint=ctx["endpoint"],
                upstream_url=attempt.state["upstream_url"],
                provider=channel_id,
                model=ctx["request_model_name"],
                actual_model=attempt.original_model,
            )

    def _lingjing_response_from_upstream(self, attempt: Any, ctx: dict[str, Any], upstream_resp: httpx.Response) -> Response:
        success = 200 <= upstream_resp.status_code < 300
        current_info = ctx["current_info"]
        channel_id = attempt.state["channel_id"]
        current_info["first_response_time"] = 0 if success else -1
        current_info["success"] = success
        current_info["provider"] = channel_id if success else None
        ctx["background_tasks"].add_task(
            update_channel_stats,
            current_info["request_id"],
            channel_id,
            ctx["request_model_name"],
            current_info["api_key"],
            success=success,
            provider_api_key=attempt.provider_api_key_raw,
        )
        if success:
            return self._raw_response(upstream_resp)
        if 400 <= upstream_resp.status_code < 500 and upstream_resp.status_code not in (408, 409, 425, 429):
            return self._raw_response(upstream_resp)
        ctx["last_error_response"].clear()
        ctx["last_error_response"].update(
            {
                "body": upstream_resp.content,
                "headers": _copy_upstream_response_headers(upstream_resp.headers),
            }
        )
        raise HTTPException(
            status_code=upstream_resp.status_code,
            detail=upstream_resp.content.decode("utf-8", errors="replace"),
        )

    def _lingjing_after_failure(self, attempt: Any, exc: Exception, status_code: int, error_message: Any, ctx: dict[str, Any]) -> None:
        current_info = ctx["current_info"]
        if attempt.state.get("track_channel_stats"):
            ctx["background_tasks"].add_task(
                update_channel_stats,
                current_info["request_id"],
                attempt.state["channel_id"],
                ctx["request_model_name"],
                current_info["api_key"],
                success=False,
                provider_api_key=attempt.provider_api_key_raw,
            )
        trace_logger.error(
            "%s upstream error status=%s error_type=%s request_id=%s model=%s provider=%s key=%s upstream_url=%s: %s",
            ctx["endpoint"],
            status_code,
            type(exc).__name__,
            ctx["request_id"],
            ctx["request_model_name"],
            attempt.state.get("channel_id", attempt.provider_name),
            _mask_secret_for_log(attempt.provider_api_key_raw),
            attempt.state.get("upstream_url", ""),
            error_message,
        )

    def _lingjing_should_cool_down(self, exc: Exception, status_code: int, error_message: Any, attempt: Any) -> bool:
        _ = exc, error_message, attempt
        return status_code in (401, 403, 429) or status_code >= 500

    def _lingjing_build_error_response(self, status_code: int, error_message: Any, ctx: dict[str, Any]):
        current_info = ctx["current_info"]
        current_info["first_response_time"] = -1
        current_info["success"] = False
        current_info["provider"] = None
        last_error_response = ctx["last_error_response"]
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

    def _lingjing_build_final_response(self, completed_plan: Any, ctx: dict[str, Any]):
        current_info = ctx["current_info"]
        current_info["first_response_time"] = -1
        current_info["success"] = False
        current_info["provider"] = None
        return JSONResponse(
            status_code=completed_plan.status_code,
            content={"error": f"All {ctx['request_model_name']} error: {completed_plan.error_message}"},
        )

model_handler = ModelRequestHandler()
responses_handler = ResponsesRequestHandler()
messages_handler = MessagesPassthroughHandler()
video_task_handler = VideoTaskHandler()
lingjing_openapi_handler = LingjingOpenapiHandler()

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = get_runtime_api_list()
    return require_api_key_index(api_list, credentials.credentials)

async def verify_admin_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_list = get_runtime_api_list()
    return require_admin_api_key(app.state.api_keys_db, api_list, credentials.credentials)

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
    return await search_response(
        model_handler=model_handler,
        http_request=request,
        background_tasks=background_tasks,
        query=q,
        api_index=api_index,
    )

@app.post("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
async def chat_completions_route(
    http_request: Request,
    request: RequestModel,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    return await chat_completions_response(
        model_handler=model_handler,
        http_request=http_request,
        request=request,
        background_tasks=background_tasks,
        api_index=api_index,
    )

@app.post("/v1/responses", dependencies=[Depends(rate_limit_dependency)])
async def responses_route(
    http_request: Request,
    request: ResponsesRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    return await responses_api_response(
        responses_handler=responses_handler,
        http_request=http_request,
        request=request,
        background_tasks=background_tasks,
        api_index=api_index,
    )

@app.post("/v1/responses/compact", dependencies=[Depends(rate_limit_dependency)])
async def responses_compact_route(
    http_request: Request,
    request: ResponsesRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    return await responses_api_response(
        responses_handler=responses_handler,
        http_request=http_request,
        request=request,
        background_tasks=background_tasks,
        api_index=api_index,
        endpoint="/v1/responses/compact",
    )

@app.post("/v1/messages", dependencies=[Depends(rate_limit_dependency)])
async def messages_route(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request: dict[str, Any] = Body(...),
    api_index: int = Depends(verify_api_key),
):
    return await messages_response(
        messages_handler=messages_handler,
        http_request=http_request,
        request_body=request,
        background_tasks=background_tasks,
        api_index=api_index,
    )

# @app.options("/v1/chat/completions", dependencies=[Depends(rate_limit_dependency)])
# async def options_handler():
#     return JSONResponse(status_code=200, content={"detail": "OPTIONS allowed"})

@app.get("/v1/models", dependencies=[Depends(rate_limit_dependency)])
async def list_models(api_index: int = Depends(verify_api_key)):
    runtime_api_list = get_runtime_api_list()
    return JSONResponse(
        content=list_models_payload(
            api_index=api_index,
            api_list=runtime_api_list,
            model_response_cache=getattr(app.state, "model_response_cache", {}) or {},
            config=app.state.config,
            models_list=app.state.models_list,
            build_models=post_all_models,
        )
    )

@app.post("/v1/images/generations", dependencies=[Depends(rate_limit_dependency)])
async def images_generations(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await image_generation_response(model_handler, request, api_index, background_tasks)

@app.post("/v1/video/tasks", dependencies=[Depends(rate_limit_dependency)])
async def video_tasks_create(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request_body: dict[str, Any] = Body(...),
    api_index: int = Depends(verify_api_key),
):
    return await video_task_create_response(
        video_task_handler=video_task_handler,
        http_request=http_request,
        request_body=request_body,
        api_index=api_index,
        background_tasks=background_tasks,
    )

@app.get("/v1/video/tasks/{task_id}", dependencies=[Depends(rate_limit_dependency)])
async def video_tasks_get(
    http_request: Request,
    task_id: str,
    background_tasks: BackgroundTasks,
    model: Optional[str] = Query(None),
    api_index: int = Depends(verify_api_key),
):
    return await video_task_get_response(
        video_task_handler=video_task_handler,
        http_request=http_request,
        task_id=task_id,
        api_index=api_index,
        background_tasks=background_tasks,
        model=model,
    )

@app.post("/v1/asset-groups", dependencies=[Depends(rate_limit_dependency)])
async def asset_groups_create(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request_body: dict[str, Any] = Body(...),
    api_index: int = Depends(verify_api_key),
):
    return await asset_groups_create_response(
        lingjing_openapi_handler=lingjing_openapi_handler,
        http_request=http_request,
        request_body=request_body,
        api_index=api_index,
        background_tasks=background_tasks,
        endpoint=VIDEO_ASSET_GROUPS_ENDPOINT,
    )

@app.get("/v1/asset-groups/{group_id}", dependencies=[Depends(rate_limit_dependency)])
async def asset_group_get(
    http_request: Request,
    group_id: str,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    return await asset_group_get_response(
        lingjing_openapi_handler=lingjing_openapi_handler,
        http_request=http_request,
        group_id=group_id,
        api_index=api_index,
        background_tasks=background_tasks,
        endpoint=VIDEO_ASSET_GROUPS_ENDPOINT,
    )

@app.post("/v1/assets", dependencies=[Depends(rate_limit_dependency)])
async def assets_create(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request_body: dict[str, Any] = Body(...),
    api_index: int = Depends(verify_api_key),
):
    return await assets_create_response(
        lingjing_openapi_handler=lingjing_openapi_handler,
        http_request=http_request,
        request_body=request_body,
        api_index=api_index,
        background_tasks=background_tasks,
        endpoint=VIDEO_ASSETS_ENDPOINT,
    )

@app.get("/v1/assets/{asset_id}", dependencies=[Depends(rate_limit_dependency)])
async def asset_get(
    http_request: Request,
    asset_id: str,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    return await asset_get_response(
        lingjing_openapi_handler=lingjing_openapi_handler,
        http_request=http_request,
        asset_id=asset_id,
        api_index=api_index,
        background_tasks=background_tasks,
        endpoint=VIDEO_ASSETS_ENDPOINT,
    )

@app.post("/v1/images/edits", dependencies=[Depends(rate_limit_dependency)])
async def images_edits(
    http_request: Request,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await image_edit_response(model_handler, http_request, api_index, background_tasks)

@app.post("/v1/embeddings", dependencies=[Depends(rate_limit_dependency)])
async def embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await embeddings_response(model_handler, request, api_index, background_tasks)

@app.post("/v1/audio/speech", dependencies=[Depends(rate_limit_dependency)])
async def audio_speech(
    request: TextToSpeechRequest,
    background_tasks: BackgroundTasks,
    api_index: str = Depends(verify_api_key)
):
    return await audio_speech_response(model_handler, request, api_index, background_tasks)

@app.post("/v1/moderations", dependencies=[Depends(rate_limit_dependency)])
async def moderations(
    request: ModerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await moderation_response(model_handler, request, api_index, background_tasks)

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
    return await audio_transcription_response(
        model_handler=model_handler,
        http_request=http_request,
        background_tasks=background_tasks,
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        api_index=api_index,
    )

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
    _ = request, token
    return await stats_summary_response(
        repository=stats_repository,
        hours=hours,
        database_disabled=DISABLE_DATABASE,
    )

@app.get("/", dependencies=[Depends(rate_limit_dependency)])
async def root():
    return RedirectResponse(url="https://uni-api-web.pages.dev", status_code=302)

# async def on_fetch(request, env):
#     import asgi
#     return await asgi.fetch(app, request, env)

@app.get("/v1/api_config", dependencies=[Depends(rate_limit_dependency)])
async def api_config(api_index: int = Depends(verify_admin_api_key)):
    _ = api_index
    return await api_config_response(app.state.config)

@app.post("/v1/api_config/update", dependencies=[Depends(rate_limit_dependency)])
async def api_config_update(api_index: int = Depends(verify_admin_api_key), config: dict = Body(...)):
    _ = api_index
    return await api_config_update_response(
        app=app,
        config_patch=config,
        update_config=update_config,
        refresh_runtime_state=refresh_runtime_state,
    )

async def query_token_usage(
    session: AsyncSession,
    filter_api_key: Optional[str] = None,
    filter_model: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None
) -> List[Dict]:
    """Queries the RequestStat table for aggregated token usage."""
    _ = session
    return await stats_repository.query_token_usage(
        filter_api_key=filter_api_key,
        filter_model=filter_model,
        start_dt=start_dt,
        end_dt=end_dt,
    )

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
    return await stats_repository.query_token_usage(
        filter_api_key=filter_api_key,
        filter_model=filter_model,
        start_dt=start_dt_obj,
        end_dt=end_dt_obj,
    )

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
    _ = request
    return await token_usage_response(
        repository=stats_repository,
        database_disabled=DISABLE_DATABASE,
        config=app.state.config,
        admin_api_keys=getattr(app.state, "admin_api_key", []),
        api_index=api_index,
        api_key_param=api_key_param,
        model=model,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        last_n_days=last_n_days,
        update_paid_key_state=lambda paid_key: update_paid_api_keys_states(app, paid_key),
    )


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
    _ = request, token
    return await channel_key_rankings_response(
        repository=stats_repository,
        database_disabled=DISABLE_DATABASE,
        provider_name=provider_name,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        last_n_days=last_n_days,
    )

@app.get("/v1/api_keys_states", dependencies=[Depends(rate_limit_dependency)])
async def api_keys_states(token: str = Depends(verify_admin_api_key)):
    _ = token
    return api_keys_states_response(app.state.paid_api_keys_states)

@app.post("/v1/add_credits", dependencies=[Depends(rate_limit_dependency)])
async def add_credits_to_api_key(
    request: Request, # Inject request to access app.state
    paid_key: str = Query(..., description="The API key to add credits to"),
    amount: float = Query(..., description="The amount of credits to add. Must be positive.", gt=0),
    token: str = Depends(verify_admin_api_key)
):
    _ = request, token
    response = add_credits_response(
        paid_api_keys_states=app.state.paid_api_keys_states,
        paid_key=paid_key,
        amount=amount,
    )
    logger.info("Credits for API key %r updated. Amount added: %s", paid_key, amount)
    return response

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
