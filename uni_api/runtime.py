import os
import re
import json
import base64
import uuid
import functools
from dataclasses import dataclass, field
import httpx
import string
import secrets
import tomllib
import asyncio
import random
from asyncio import Semaphore
from time import perf_counter_ns, time, time_ns
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter, defaultdict
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
from uni_api.routing.request_rules import request_reasoning_effort
from uni_api.routing.request_types import detect_request_type
from upstream import (
    UPSTREAM_NETWORK_ERRORS,
    UpstreamRunner,
    build_upstream_error_response,
    finalize_latest_routing_attempt,
    finalize_response_memory_attempt,
    finalize_routing_attempt,
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
from uni_api.api.alpha_search import (
    ALPHA_SEARCH_ENDPOINT,
    AlphaSearchRequestHandler,
)
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
from uni_api.api.models import (
    CODEX_PRO_MODELS_SNAPSHOT_CLIENT_VERSION,
    CODEX_PRO_MODELS_SNAPSHOT_UPSTREAM_ETAG,
    codex_models_payload,
    list_models_payload,
)
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
from uni_api.rust_responses_snapshot import publish_rust_responses_snapshot
from uni_api.http_content import is_json_media_type
from uni_api.idempotency import apply_oaix_routing_attempt_id
from uni_api.observability.paid_keys import compute_paid_api_key_state
from uni_api.observability.request_context import (
    get_request_info,
    request_info,
)
from uni_api.observability.spans import merge_timing_spans
from uni_api.observability.telemetry import (
    emit_admission_503_response_write_outcome,
    emit_large_body_admission_decision,
    emit_request_observability,
    emit_response_buffer_event,
    emit_terminal_hop_observation,
    emit_worker_cpu_profile,
    emit_worker_runtime_snapshot,
    observability_exporter_snapshot,
)
from uni_api.observability.responses_stream import (
    ObservedResponseByteIterator,
    ResponsesStreamDiagnostics,
    is_oaix_terminal_flush_marker,
    observe_pool_sweeper_connection_close,
)
from uni_api.observability.worker_runtime import WorkerRuntimeObserver
from uni_api.observability.threadpool_tasks import submit_threadpool_task
from uni_api.observability.middleware import (
    StatsMiddleware,
    StatsMiddlewareDependencies,
)
from uni_api.admission import (
    AdaptiveNetworkGovernor,
    Admission503ResponseWriteOutcome,
    AdmissionRejected,
    RequestAdmissionController,
    get_request_admission_lease,
)
from uni_api.admission.cpu import cpu_phase_snapshot
from uni_api.admission.json_parsing import (
    JSON_PARSE_CPU_WORKERS,
    ReusableJSONParseWorkspace,
    run_json_cpu,
)
from uni_api.admission.json_memory import DEFAULT_JSON_RAW_MEMORY_MULTIPLIER
from uni_api.admission.memory import process_memory_governor
from uni_api.admission.resources import (
    kernel_somaxconn,
    startup_concurrency_from_environment,
    startup_large_request_memory_limit,
    startup_per_request_memory_limit,
)
from uni_api.middleware.admission import (
    ADMISSION_LEASE_STATE_KEY,
    ADMISSION_REQUEST_ID_STATE_KEY,
    ADMISSION_TRACE_ID_STATE_KEY,
    RequestAdmissionMiddleware,
)
from uni_api.middleware.idempotency import (
    IdempotencyMiddleware,
    build_default_idempotency_coordinator,
)
from uni_api.routing.health import ProviderModelCircuitBreaker
from uni_api.middleware.request_decompression import (
    REQUEST_BODY_CPU_WORKERS,
    REQUEST_BODY_COMPLEXITY_INFO_KEY,
    RequestBodyDecompressionMiddleware,
    mark_request_body_releasable_at_response_start,
    request_body_response_started,
    request_body_complexity_diagnostics_from_scope,
)
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
from uni_api.streaming.logging_response import (
    LoggingStreamingResponse,
    timed_out_io_task_snapshot,
)
from uni_api.streaming.error_text import bounded_stream_error_text
from uni_api.streaming.chat_completion_collector import (
    collect_openai_chat_completion_from_streaming_sse,
)
from uni_api.streaming.bounded_queue import (
    ByteBoundedQueue,
    ObservedStreamFrame,
    ObservedStreamChunk,
    ReservedChunkBuffer,
    ReservedStreamChunk,
    RetainedByteBudget,
    StreamBufferBudgetTimeout,
    StreamQueueClosed,
    StreamQueueItemLease,
    StreamQueueItemTooLarge,
    StreamQueuePutTimeout,
)
from uni_api.streaming.sse import (
    DEFAULT_MAX_EVENT_BYTES,
    IncrementalSSEParser,
    SSEProtocolError,
    StreamParserRetainedLease,
    UNLIMITED_SSE_BYTES,
    is_sse_comment_frame,
    parse_owned_sse_event,
    parse_sse_event,
    sse_event_has_data_field,
    stream_parser_retained_budget_snapshot,
    validate_sse_event_type_consistency,
)
from uni_api.streaming.responses_fast_path import (
    CanonicalResponsesDeltaFrame,
    MIN_CANONICAL_RESPONSES_DELTA_FRAME_BYTES,
    can_forward_responses_delta_without_materializing,
    match_canonical_responses_delta_frame,
)
from uni_api.streaming.usage import (
    StreamUsageSnapshot,
    stream_usage_snapshot,
    stream_usage_snapshot_from_payload,
)
from uni_api.server import build_bounded_h11_protocol
from uni_api.rust_responses_control import (
    RustResponsesControlError,
    RustResponsesControlPlane,
    RustResponsesResolvedOutcome,
)
from uni_api.upstream.client_pool import ClientPool
from uni_api.upstream.request_replay import MessagesRequestReplayStore
from uni_api.upstream.response_limits import (
    UPSTREAM_RESPONSE_CPU_WORKERS,
    read_limited_response_body,
)
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
from uni_api.upstream.responses_errors import (
    ResponsesSemanticError,
    responses_failure_error,
)
from uni_api.upstream.responses_normalization import (
    ResponsesCustomToolCallIdCollisionError,
    ResponsesCustomToolCallIdNormalizationResult,
    ResponsesCustomToolCallIdNormalizer,
    responses_custom_tool_call_id_normalization_enabled,
)
from uni_api.upstream.transport_errors import (
    TransportErrorClassification,
    classify_httpx_transport_error,
)
from core.utils import safe_get

from sqlalchemy import inspect, text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def _bounded_env_int(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        value = int(default)
    else:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > maximum:
        raise ValueError(
            f"{name}={value} exceeds the startup safety limit {maximum}"
        )
    return value


def _positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        value = int(default)
    else:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


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
RUST_RESPONSES_CONTROL_HEADER = "x-uni-api-rust-control-token"
RUST_RESPONSES_SESSION_HEADER = "x-uni-api-rust-responses-session"
RUST_RESPONSES_INTERNAL_PREFIX = "/_internal/rust-responses"
RUST_REQUEST_SPOOL_INT_HEADERS = {
    "x-uni-api-rust-request-spool-body-bytes": "rust_request_spool_body_bytes",
    "x-uni-api-rust-request-spool-memory-peak-bytes": "rust_request_spool_memory_peak_bytes",
    "x-uni-api-rust-request-spool-local-disk-bytes": "rust_request_spool_local_disk_bytes",
    "x-uni-api-rust-request-spool-local-free-bytes-start": "rust_request_spool_local_free_bytes_start",
    "x-uni-api-rust-request-spool-local-writable-bytes-start": "rust_request_spool_local_writable_bytes_start",
    "x-uni-api-rust-request-spool-local-free-inodes-start": "rust_request_spool_local_free_inodes_start",
    "x-uni-api-rust-request-spool-local-writable-inodes-start": "rust_request_spool_local_writable_inodes_start",
    "x-uni-api-rust-request-spool-resource-wait-ms": "rust_request_spool_resource_wait_ms",
}
RUST_REQUEST_SPOOL_TIER_HEADER = "x-uni-api-rust-request-spool-final-tier"
RUST_RESPONSES_CONTROL_TOKEN = os.getenv(
    "UNI_API_RUST_CONTROL_TOKEN",
    "",
).strip()
RUST_RESPONSES_DATA_PLANE_ENABLED = bool(
    RUST_RESPONSES_CONTROL_TOKEN
    and _env_bool("UNI_API_RUST_RESPONSES_DATA_PLANE", True)
)
# The downstream usage observer is deliberately best-effort and bounded. It
# may disable itself for a larger Responses event without affecting proxying.
RESPONSES_CANONICAL_EVENT_HEADER_MAX_BYTES = len(b"event: ") + 256 + len(b"\n")
RESPONSES_CANONICAL_EVENT_MAX_BYTES = (
    DEFAULT_MAX_EVENT_BYTES + RESPONSES_CANONICAL_EVENT_HEADER_MAX_BYTES
)
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
        self.tcp_states: dict[str, int] = {}
        self.upstream_pool_in_use = 0
        self.upstream_pool_wait_ms = 0
        self._request_admission_snapshot: Optional[Callable[[], dict[str, Any]]] = None
        self._upstream_client_snapshot: Optional[Callable[[], dict[str, Any]]] = None
        self._stream_byte_budget_snapshot: Optional[Callable[[], Any]] = None
        self._stream_parser_budget_snapshot: Optional[
            Callable[[], dict[str, int]]
        ] = None
        self._stream_stats_snapshot: Optional[Callable[[], dict[str, Any]]] = None
        self._memory_parent_snapshot: Optional[Callable[[], Any]] = None
        self._observability_exporter_snapshot: Optional[
            Callable[[], dict[str, int]]
        ] = None
        self._worker_runtime_snapshot: Optional[
            Callable[[], dict[str, Any]]
        ] = None
        self._network_sampler_task: Optional[asyncio.Task[None]] = None
        self._stream_queues: dict[int, ByteBoundedQueue] = {}
        self._retired_stream_queue_blocked_puts = 0
        self._retired_stream_queue_put_wait_ms = 0.0
        self._retired_stream_queue_put_timeouts = 0
        self._admission_503_response_write_completed: Counter[str] = Counter()
        self._admission_503_response_write_failed: Counter[str] = Counter()
        self._response_budget_alert_state = (False, False)
        self._response_budget_alert_last_logged_at = 0.0

    def attach_request_admission(
        self,
        snapshot: Callable[[], dict[str, Any]],
    ) -> None:
        self._request_admission_snapshot = snapshot

    def attach_upstream_client(
        self,
        snapshot: Callable[[], dict[str, Any]],
    ) -> None:
        self._upstream_client_snapshot = snapshot

    def attach_stream_byte_budget(self, snapshot: Callable[[], Any]) -> None:
        self._stream_byte_budget_snapshot = snapshot

    def attach_stream_parser_budget(
        self,
        snapshot: Callable[[], dict[str, int]],
    ) -> None:
        self._stream_parser_budget_snapshot = snapshot

    def attach_stream_stats(self, snapshot: Callable[[], dict[str, Any]]) -> None:
        self._stream_stats_snapshot = snapshot

    def attach_memory_parent(self, snapshot: Callable[[], Any]) -> None:
        self._memory_parent_snapshot = snapshot

    def attach_observability_exporter(
        self,
        snapshot: Callable[[], dict[str, int]],
    ) -> None:
        self._observability_exporter_snapshot = snapshot

    def attach_worker_runtime(
        self,
        snapshot: Callable[[], dict[str, Any]],
    ) -> None:
        self._worker_runtime_snapshot = snapshot


    def register_stream_queue(self, queue: ByteBoundedQueue) -> None:
        self._stream_queues[id(queue)] = queue

    def unregister_stream_queue(self, queue: ByteBoundedQueue) -> None:
        registered = self._stream_queues.pop(id(queue), None)
        if registered is None:
            return
        snapshot = registered.snapshot()
        self._retired_stream_queue_blocked_puts += snapshot.blocked_puts
        self._retired_stream_queue_put_wait_ms += snapshot.put_wait_ms
        self._retired_stream_queue_put_timeouts += snapshot.put_timeouts

    def begin_inflight(self) -> None:
        self.inflight_requests += 1

    def end_inflight(self) -> None:
        self.inflight_requests = max(0, self.inflight_requests - 1)

    def record_admission_503_response_write(
        self,
        reason: str,
        *,
        completed: bool,
    ) -> tuple[int, int]:
        normalized = str(reason or "unknown").strip() or "unknown"
        target = (
            self._admission_503_response_write_completed
            if completed
            else self._admission_503_response_write_failed
        )
        target[normalized] += 1
        return (
            sum(self._admission_503_response_write_completed.values()),
            sum(self._admission_503_response_write_failed.values()),
        )

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

    def record_upstream_pool_wait(self, wait_ms: float) -> None:
        self.upstream_pool_wait_ms = max(0, int(round(float(wait_ms))))

    async def record_event_loop_lag(self) -> None:
        started_at = time()
        await asyncio.sleep(0)
        self.event_loop_lag_ms = int((time() - started_at) * 1000)

    async def start_network_sampler(self, *, interval_seconds: float = 5.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        if self._network_sampler_task is not None:
            return
        await self._sample_network_state()
        self._network_sampler_task = asyncio.create_task(
            self._network_sampler_loop(interval_seconds),
            name="uni-api-runtime-network-sampler",
        )

    async def stop_network_sampler(self) -> None:
        if self._network_sampler_task is None:
            return
        self._network_sampler_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._network_sampler_task
        self._network_sampler_task = None

    async def _network_sampler_loop(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            await self._sample_network_state()

    async def _sample_network_state(self) -> None:
        ticket = submit_threadpool_task("network_procfs")
        try:
            open_sockets, tcp_states = await asyncio.to_thread(
                ticket.run,
                lambda: (_open_socket_count(), _tcp_state_counts()),
            )
        except asyncio.CancelledError:
            ticket.cancel_if_queued()
            raise
        self.open_sockets = open_sockets
        self.tcp_states = tcp_states

    def _observe_response_budget_alerts(self, admission: dict[str, Any]) -> None:
        headroom_alert = bool(
            admission.get("response_budget_soft_headroom_alert")
        )
        rejection_alert = bool(admission.get("response_rejection_rate_alert"))
        state = (headroom_alert, rejection_alert)
        previous = self._response_budget_alert_state
        now = time()
        if any(state) and (
            state != previous
            or now - self._response_budget_alert_last_logged_at >= 60.0
        ):
            logger.warning(
                "Response admission budget alert headroom_alert=%s "
                "rejection_rate_alert=%s soft_remaining_bytes=%s "
                "soft_remaining_ratio=%s rejections_1m=%s branches=%s",
                headroom_alert,
                rejection_alert,
                admission.get("response_budget_soft_remaining_bytes"),
                admission.get("response_budget_soft_remaining_ratio"),
                admission.get("response_rejections_1m"),
                admission.get("response_rejection_decisions_by_branch") or {},
            )
            self._response_budget_alert_last_logged_at = now
        elif any(previous) and not any(state):
            logger.info(
                "Response admission budget alert recovered "
                "soft_remaining_bytes=%s rejections_1m=%s",
                admission.get("response_budget_soft_remaining_bytes"),
                admission.get("response_rejections_1m"),
            )
        self._response_budget_alert_state = state

    def snapshot(self) -> dict[str, Any]:
        admission: dict[str, Any] = {}
        if self._request_admission_snapshot is not None:
            admission = self._request_admission_snapshot()
        request_active = int(admission.get("active", self.inflight_requests) or 0)
        request_waiters = int(admission.get("waiters", 0) or 0)
        rejected = admission.get("rejected")
        if not isinstance(rejected, dict):
            rejected = {}
        self._observe_response_budget_alerts(admission)

        upstream_admission: dict[str, Any] = {}
        if self._upstream_client_snapshot is not None:
            upstream_client = self._upstream_client_snapshot()
            candidate = upstream_client.get("admission")
            if isinstance(candidate, dict):
                upstream_admission = candidate

        queue_snapshots = [queue.snapshot() for queue in self._stream_queues.values()]
        stream_budget: Any = None
        if self._stream_byte_budget_snapshot is not None:
            stream_budget = self._stream_byte_budget_snapshot()
        stream_parser_budget: dict[str, int] = {}
        if self._stream_parser_budget_snapshot is not None:
            stream_parser_budget = self._stream_parser_budget_snapshot()
        stream_stats: dict[str, Any] = {}
        if self._stream_stats_snapshot is not None:
            stream_stats = self._stream_stats_snapshot()
        timed_out_io = timed_out_io_task_snapshot()
        memory_parent: Any = None
        if self._memory_parent_snapshot is not None:
            memory_parent = self._memory_parent_snapshot()
        observability_exporter: dict[str, int] = {}
        if self._observability_exporter_snapshot is not None:
            observability_exporter = self._observability_exporter_snapshot()
        worker_runtime: dict[str, Any] = {}
        if self._worker_runtime_snapshot is not None:
            worker_runtime = self._worker_runtime_snapshot()
        cpu_phase = cpu_phase_snapshot()
        stream_queue_blocked_puts = self._retired_stream_queue_blocked_puts + sum(
            snapshot.blocked_puts for snapshot in queue_snapshots
        )
        stream_queue_put_wait_ms = self._retired_stream_queue_put_wait_ms + sum(
            snapshot.put_wait_ms for snapshot in queue_snapshots
        )
        stream_queue_put_timeouts = self._retired_stream_queue_put_timeouts + sum(
            snapshot.put_timeouts for snapshot in queue_snapshots
        )
        return {
            "service": "uni-api-ember",
            "worker_runtime": worker_runtime,
            **worker_runtime,
            # Keep the established name, but source it from the outer ASGI
            # lease so streaming requests remain active until their final byte.
            "inflight_requests": request_active,
            "request_active": request_active,
            "request_waiters": request_waiters,
            "request_capacity": admission.get("capacity"),
            "request_admission_mode": admission.get(
                "mode",
                RESOURCE_ADMISSION_MODE,
            ),
            "request_admission_sizing_source": (
                REQUEST_ADMISSION_ACTIVE_SIZING_SOURCE
            ),
            "request_control_memory_bytes_per_request": admission.get(
                "control_memory_bytes_per_request"
            ),
            "request_control_memory_reserved_bytes": admission.get(
                "control_memory_reserved_bytes"
            ),
            "cpu_phase": cpu_phase,
            "cpu_phase_capacity": cpu_phase.get("capacity"),
            "cpu_phase_active": cpu_phase.get("active"),
            "cpu_phase_waiters": cpu_phase.get("waiters"),
            "request_startup_cpu_millicores": REQUEST_ADMISSION_CPU_MILLICORES,
            "request_startup_cpu_weight": REQUEST_ADMISSION_CPU_WEIGHT,
            "request_startup_cpu_affinity_count": (
                REQUEST_ADMISSION_CPU_AFFINITY_COUNT
            ),
            "request_startup_cpu_sizing_source": (
                REQUEST_ADMISSION_CPU_SIZING_SOURCE
            ),
            "request_startup_cpu_active_limit": (
                REQUEST_ADMISSION_CPU_ACTIVE_LIMIT
                if RESOURCE_ADMISSION_MODE == "legacy"
                else None
            ),
            "request_startup_resource_active_limit": (
                REQUEST_ADMISSION_RESOURCE_ACTIVE_LIMIT
                if RESOURCE_ADMISSION_MODE == "legacy"
                else None
            ),
            "request_startup_memory_available_bytes": (
                REQUEST_ADMISSION_STARTUP_MEMORY_AVAILABLE_BYTES
            ),
            "request_startup_control_memory_budget_bytes": (
                REQUEST_ADMISSION_CONTROL_MEMORY_BUDGET_BYTES
            ),
            "request_startup_nofile_soft_limit": (
                REQUEST_ADMISSION_NOFILE_SOFT_LIMIT
            ),
            "request_startup_open_fds": REQUEST_ADMISSION_STARTUP_OPEN_FDS,
            "request_startup_ephemeral_port_count": (
                REQUEST_ADMISSION_EPHEMERAL_PORT_COUNT
            ),
            "request_startup_ephemeral_port_occupancy": (
                REQUEST_ADMISSION_EPHEMERAL_PORT_OCCUPANCY
            ),
            "request_startup_ephemeral_active_limit": (
                REQUEST_ADMISSION_EPHEMERAL_ACTIVE_LIMIT
            ),
            "request_waiter_limit": admission.get("waiter_limit"),
            "request_wait_timeout_seconds": admission.get(
                "wait_timeout_seconds"
            ),
            "request_total_limit": (
                REQUEST_ADMISSION_TOTAL_LIMIT
                if RESOURCE_ADMISSION_MODE == "legacy"
                else None
            ),
            "uvicorn_limit_concurrency": None,
            "uvicorn_connection_limit": UVICORN_CONNECTION_LIMIT,
            "uvicorn_backlog": UVICORN_BACKLOG,
            "uvicorn_http_protocol": BOUNDED_HTTP_PROTOCOL_STATS.snapshot(),
            # Admission charges conservative in-memory weights (for example
            # JSON raw bytes x5 plus structural token charges), not raw wire
            # bytes. Keep that semantic explicit
            # so dashboards do not mislabel the values.
            "request_body_reserved_weighted_bytes": admission.get(
                "reserved_body_bytes"
            ),
            "runtime_global_request_body_reserved_weighted_bytes": admission.get(
                "reserved_body_bytes"
            ),
            "upstream_response_reserved_weighted_bytes": admission.get(
                "reserved_response_bytes"
            ),
            "runtime_global_upstream_response_reserved_weighted_bytes": admission.get(
                "reserved_response_bytes"
            ),
            "request_retained_reserved_weighted_bytes": admission.get(
                "reserved_retained_bytes"
            ),
            "runtime_global_retained_reserved_weighted_bytes": admission.get(
                "reserved_retained_bytes"
            ),
            "request_body_budget_bytes": admission.get("body_budget"),
            "runtime_global_request_body_budget_weighted_bytes": admission.get("body_budget"),
            "request_body_budget_hard_bytes": admission.get("body_budget_hard"),
            "runtime_global_request_body_budget_hard_weighted_bytes": admission.get(
                "body_budget_hard"
            ),
            "request_max_body_reserved_weighted_bytes": admission.get(
                "max_body_bytes"
            ),
            "request_max_response_reserved_weighted_bytes": admission.get(
                "max_response_bytes"
            ),
            "request_max_retained_weighted_bytes": admission.get(
                "max_retained_bytes_per_request"
            ),
            "request_wire_body_max_bytes": REQUEST_WIRE_BODY_MAX_BYTES,
            "request_product_wire_body_max_bytes": (
                _product_request_wire_body_max_bytes
            ),
            "request_json_complexity_max_bytes": (
                REQUEST_JSON_COMPLEXITY_MAX_BYTES
            ),
            "request_large_body_threshold_weighted_bytes": admission.get(
                "large_body_threshold_weighted_bytes"
            ),
            "runtime_global_large_body_threshold_weighted_bytes": admission.get(
                "large_body_threshold_weighted_bytes"
            ),
            "request_large_body_limit": admission.get("large_body_limit"),
            "runtime_global_large_body_limit": admission.get("large_body_limit"),
            "request_large_body_active": admission.get("large_body_active"),
            "runtime_global_large_body_active": admission.get("large_body_active"),
            "runtime_global_large_body_oldest_holder_age_ms": admission.get(
                "large_body_oldest_holder_age_ms"
            ),
            "json_parse_cpu_workers": JSON_PARSE_CPU_WORKERS,
            "request_body_cpu_workers": REQUEST_BODY_CPU_WORKERS,
            "upstream_response_cpu_workers": UPSTREAM_RESPONSE_CPU_WORKERS,
            "request_retained_budget_bytes": admission.get("body_budget"),
            "request_deferred_memory_requests": admission.get(
                "deferred_memory_requests"
            ),
            "runtime_global_deferred_memory_requests": admission.get(
                "deferred_memory_requests"
            ),
            "request_deferred_memory_weighted_bytes": admission.get(
                "deferred_memory_bytes"
            ),
            "runtime_global_deferred_memory_weighted_bytes": admission.get(
                "deferred_memory_bytes"
            ),
            "request_admission_rejected": dict(rejected),
            "request_admission_rejected_total": sum(
                int(value or 0) for value in rejected.values()
            ),
            "runtime_global_admission_rejection_decisions": dict(rejected),
            "runtime_global_admission_rejection_decision_total": int(
                admission.get("rejection_decision_total")
                or sum(int(value or 0) for value in rejected.values())
            ),
            "runtime_global_response_admission_rejections_by_branch": dict(
                admission.get("response_rejection_decisions_by_branch") or {}
            ),
            "runtime_global_response_admission_rejection_total": admission.get(
                "response_rejection_decision_total"
            ),
            "runtime_global_response_admission_rejections_1m": admission.get(
                "response_rejections_1m"
            ),
            "runtime_global_response_admission_rejection_rate_per_second_1m": admission.get(
                "response_rejection_rate_per_second_1m"
            ),
            "runtime_global_response_budget_soft_remaining_bytes": admission.get(
                "response_budget_soft_remaining_bytes"
            ),
            "runtime_global_response_budget_soft_remaining_ratio": admission.get(
                "response_budget_soft_remaining_ratio"
            ),
            "runtime_global_response_budget_soft_headroom_alert": admission.get(
                "response_budget_soft_headroom_alert"
            ),
            "runtime_global_response_rejection_rate_alert": admission.get(
                "response_rejection_rate_alert"
            ),
            "runtime_global_response_buffer_events_recorded_total": admission.get(
                "response_buffer_events_recorded_total"
            ),
            "runtime_global_response_buffer_event_history_overwritten_total": admission.get(
                "response_buffer_event_history_overwritten_total"
            ),
            "runtime_global_response_buffer_event_record_failures_total": admission.get(
                "response_buffer_event_record_failures_total"
            ),
            "runtime_global_response_buffer_event_observer_errors_total": admission.get(
                "response_buffer_event_observer_errors_total"
            ),
            "runtime_global_response_buffer_event_observer_enqueue_failures_total": admission.get(
                "response_buffer_event_observer_enqueue_failures_total"
            ),
            "runtime_global_admission_503_response_write_completed": dict(
                self._admission_503_response_write_completed
            ),
            "runtime_global_admission_503_response_write_completed_total": sum(
                self._admission_503_response_write_completed.values()
            ),
            "runtime_global_admission_503_response_write_failed": dict(
                self._admission_503_response_write_failed
            ),
            "runtime_global_admission_503_response_write_failed_total": sum(
                self._admission_503_response_write_failed.values()
            ),
            "runtime_global_large_body_decision_events_recorded_total": admission.get(
                "large_body_decision_events_recorded_total"
            ),
            "runtime_global_large_body_decision_history_overwritten_total": admission.get(
                "large_body_decision_history_overwritten_total"
            ),
            "runtime_global_large_body_decision_record_failures_total": admission.get(
                "large_body_decision_record_failures_total"
            ),
            "runtime_global_large_body_decision_observer_errors_total": admission.get(
                "large_body_decision_observer_errors_total"
            ),
            "runtime_global_large_body_decision_observer_enqueue_failures_total": admission.get(
                "large_body_decision_observer_enqueue_failures_total"
            ),
            "runtime_global_large_body_decision_export_enqueued_total": observability_exporter.get(
                "large_body_decision_enqueued_total"
            ),
            "runtime_global_large_body_decision_export_enqueue_dropped_total": observability_exporter.get(
                "large_body_decision_enqueue_dropped_total"
            ),
            "runtime_global_large_body_decision_export_build_errors_total": observability_exporter.get(
                "large_body_decision_build_errors_total"
            ),
            "runtime_global_large_body_decision_export_errors_total": observability_exporter.get(
                "large_body_decision_export_errors_total"
            ),
            "runtime_global_admission_503_outcome_export_enqueued_total": observability_exporter.get(
                "admission_503_outcome_enqueued_total"
            ),
            "runtime_global_admission_503_outcome_export_enqueue_dropped_total": observability_exporter.get(
                "admission_503_outcome_enqueue_dropped_total"
            ),
            "runtime_global_admission_503_outcome_export_build_errors_total": observability_exporter.get(
                "admission_503_outcome_build_errors_total"
            ),
            "runtime_global_admission_503_outcome_export_errors_total": observability_exporter.get(
                "admission_503_outcome_export_errors_total"
            ),
            "runtime_global_response_buffer_event_export_enqueued_total": observability_exporter.get(
                "response_buffer_event_enqueued_total"
            ),
            "runtime_global_response_buffer_event_export_enqueue_dropped_total": observability_exporter.get(
                "response_buffer_event_enqueue_dropped_total"
            ),
            "runtime_global_response_buffer_event_export_build_errors_total": observability_exporter.get(
                "response_buffer_event_build_errors_total"
            ),
            "runtime_global_response_buffer_event_export_errors_total": observability_exporter.get(
                "response_buffer_event_export_errors_total"
            ),
            "runtime_global_worker_runtime_snapshot_export_enqueued_total": observability_exporter.get(
                "worker_runtime_snapshot_enqueued_total"
            ),
            "runtime_global_worker_runtime_snapshot_export_enqueue_dropped_total": observability_exporter.get(
                "worker_runtime_snapshot_enqueue_dropped_total"
            ),
            "runtime_global_worker_runtime_snapshot_export_build_errors_total": observability_exporter.get(
                "worker_runtime_snapshot_build_errors_total"
            ),
            "runtime_global_worker_runtime_snapshot_export_errors_total": observability_exporter.get(
                "worker_runtime_snapshot_export_errors_total"
            ),
            "runtime_global_worker_cpu_profile_export_enqueued_total": observability_exporter.get(
                "worker_cpu_profile_enqueued_total"
            ),
            "runtime_global_worker_cpu_profile_export_enqueue_dropped_total": observability_exporter.get(
                "worker_cpu_profile_enqueue_dropped_total"
            ),
            "runtime_global_worker_cpu_profile_export_build_errors_total": observability_exporter.get(
                "worker_cpu_profile_build_errors_total"
            ),
            "runtime_global_worker_cpu_profile_export_errors_total": observability_exporter.get(
                "worker_cpu_profile_export_errors_total"
            ),
            "runtime_global_terminal_hop_observation_export_enqueued_total": observability_exporter.get(
                "terminal_hop_observation_enqueued_total"
            ),
            "runtime_global_terminal_hop_observation_export_enqueue_dropped_total": observability_exporter.get(
                "terminal_hop_observation_enqueue_dropped_total"
            ),
            "runtime_global_terminal_hop_observation_export_build_errors_total": observability_exporter.get(
                "terminal_hop_observation_build_errors_total"
            ),
            "runtime_global_terminal_hop_observation_export_errors_total": observability_exporter.get(
                "terminal_hop_observation_export_errors_total"
            ),
            "middleware_inflight_requests": self.inflight_requests,
            "waiting_first_byte": len(self.waiting_first_byte_requests) + self.waiting_first_byte_untracked,
            "event_loop_lag_ms": self.event_loop_lag_ms,
            "cgroup_memory_source": getattr(memory_parent, "source", None),
            "cgroup_memory_current_bytes": getattr(
                memory_parent, "current_bytes", None
            ),
            "cgroup_memory_limit_bytes": getattr(
                memory_parent, "limit_bytes", None
            ),
            "cgroup_memory_high_bytes": getattr(memory_parent, "high_bytes", None),
            "memory_soft_limit_bytes": getattr(
                memory_parent, "soft_limit_bytes", None
            ),
            "memory_guard_bytes": getattr(memory_parent, "guard_bytes", None),
            "memory_parent_capacity_bytes": getattr(
                memory_parent, "capacity_bytes", None
            ),
            "memory_parent_available_bytes": getattr(
                memory_parent, "available_bytes", None
            ),
            "memory_parent_reserved_bytes": getattr(
                memory_parent, "reserved_bytes", None
            ),
            "memory_parent_peak_reserved_bytes": getattr(
                memory_parent, "peak_reserved_bytes", None
            ),
            "memory_parent_sample_sequence": getattr(
                memory_parent, "sample_sequence", None
            ),
            "memory_parent_sample_age_ms": getattr(
                memory_parent, "sample_age_ms", None
            ),
            "memory_parent_reservations": getattr(
                memory_parent, "reservations", {}
            ),
            "memory_parent_shared_reservations": getattr(
                memory_parent, "shared_reservations", {}
            ),
            "memory_parent_rejected": getattr(memory_parent, "rejected", {}),
            "memory_parent_blocked_reservations": getattr(
                memory_parent, "blocked_reservations", None
            ),
            "memory_parent_waiting_reservations": getattr(
                memory_parent, "waiting_reservations", None
            ),
            "memory_parent_wait_timeouts": getattr(
                memory_parent, "wait_timeouts", None
            ),
            "memory_events": getattr(memory_parent, "events", {}),
            "memory_sample_error": getattr(memory_parent, "sample_error", None),
            "open_sockets": self.open_sockets,
            "tcp_states": dict(self.tcp_states),
            "tcp_close_wait": self.tcp_states.get("CLOSE_WAIT", 0),
            "upstream_pool_in_use": int(
                upstream_admission.get("active", self.upstream_pool_in_use) or 0
            ),
            "upstream_pool_waiters": int(
                upstream_admission.get("waiters", 0) or 0
            ),
            "upstream_pool_wait_ms": self.upstream_pool_wait_ms,
            "upstream_pool_wait_ms_avg": upstream_admission.get("wait_ms_avg"),
            "upstream_pool_wait_ms_max": upstream_admission.get("wait_ms_max"),
            "upstream_pool_rejected": upstream_admission.get("rejected", {}),
            "upstream_admission_mode": upstream_admission.get("mode"),
            "outbound_network_resources": upstream_admission.get(
                "network_resources",
                {},
            ),
            "stream_queue_active": len(queue_snapshots),
            "stream_queue_items": sum(snapshot.items for snapshot in queue_snapshots),
            "stream_queue_bytes": sum(snapshot.bytes for snapshot in queue_snapshots),
            "stream_queue_waiting_putters": sum(
                snapshot.waiting_putters for snapshot in queue_snapshots
            ),
            "stream_queue_peak_items": sum(
                snapshot.peak_items for snapshot in queue_snapshots
            ),
            "stream_queue_peak_bytes": sum(
                snapshot.peak_bytes for snapshot in queue_snapshots
            ),
            "stream_queue_blocked_puts": stream_queue_blocked_puts,
            "stream_queue_put_wait_ms": int(round(stream_queue_put_wait_ms)),
            "stream_queue_put_timeouts": stream_queue_put_timeouts,
            "stream_buffer_reserved_bytes": getattr(
                stream_budget, "used_bytes", None
            ),
            "stream_buffer_budget_bytes": getattr(
                stream_budget, "capacity_bytes", None
            ),
            "stream_buffer_budget_waiters": getattr(
                stream_budget, "waiting_reservations", None
            ),
            "stream_buffer_budget_peak_bytes": getattr(
                stream_budget, "peak_bytes", None
            ),
            "stream_buffer_budget_timeouts": getattr(
                stream_budget, "timeouts", None
            ),
            "stream_parser_reserved_bytes": stream_parser_budget.get(
                "used_bytes"
            ),
            "stream_parser_budget_bytes": stream_parser_budget.get(
                "capacity_bytes"
            ),
            "stream_parser_peak_bytes": stream_parser_budget.get("peak_bytes"),
            "stream_parser_rejected_total": stream_parser_budget.get("rejected"),
            "stream_stats_queue_items": stream_stats.get("items"),
            "stream_stats_queue_capacity": stream_stats.get("capacity"),
            "stream_stats_workers": stream_stats.get("workers"),
            "stream_stats_submitted": stream_stats.get("submitted"),
            "stream_stats_completed": stream_stats.get("completed"),
            "stream_stats_failed": stream_stats.get("failed"),
            "stream_stats_dropped": stream_stats.get("dropped"),
            "channel_stats_queue_items": stream_stats.get("items"),
            "channel_stats_queue_capacity": stream_stats.get("capacity"),
            "channel_stats_workers": stream_stats.get("workers"),
            "channel_stats_submitted": stream_stats.get("submitted"),
            "channel_stats_completed": stream_stats.get("completed"),
            "channel_stats_failed": stream_stats.get("failed"),
            "channel_stats_dropped": stream_stats.get("dropped"),
            "timed_out_io_tasks": timed_out_io.get("pending"),
            "timed_out_io_task_capacity": timed_out_io.get("capacity"),
        }


runtime_gauges = RuntimeGauges()
runtime_gauges.attach_memory_parent(process_memory_governor.snapshot)
runtime_gauges.attach_observability_exporter(observability_exporter_snapshot)
worker_runtime_observer = WorkerRuntimeObserver(
    inflight_supplier=lambda: runtime_gauges.inflight_requests,
    snapshot_emitter=emit_worker_runtime_snapshot,
    profile_emitter=emit_worker_cpu_profile,
    terminal_hop_emitter=emit_terminal_hop_observation,
)
runtime_gauges.attach_worker_runtime(worker_runtime_observer.snapshot)

_startup_memory_snapshot = process_memory_governor.snapshot(force=True)
_startup_memory_available = (
    _startup_memory_snapshot.available_bytes
    if _startup_memory_snapshot.sample_error is None
    else None
)
# ``weighted`` is the production default: request counts remain observable but
# never participate in admission. ``legacy`` is an emergency rollback only.
RESOURCE_ADMISSION_MODE = str(
    os.getenv("RESOURCE_ADMISSION_MODE", "weighted") or "weighted"
).strip().lower()
if RESOURCE_ADMISSION_MODE not in {"weighted", "legacy"}:
    raise ValueError("RESOURCE_ADMISSION_MODE must be weighted or legacy")
STARTUP_CONCURRENCY_ENVELOPE = startup_concurrency_from_environment(
    memory_available_bytes=_startup_memory_available,
    honor_request_count_overrides=RESOURCE_ADMISSION_MODE == "legacy",
)
REQUEST_ADMISSION_ACTIVE_LIMIT = STARTUP_CONCURRENCY_ENVELOPE.active_limit
REQUEST_ADMISSION_WAITER_LIMIT = STARTUP_CONCURRENCY_ENVELOPE.waiter_limit
REQUEST_ADMISSION_TOTAL_LIMIT = STARTUP_CONCURRENCY_ENVELOPE.total_limit
REQUEST_ADMISSION_ACTIVE_SIZING_SOURCE = (
    "weighted_resources"
    if RESOURCE_ADMISSION_MODE == "weighted"
    else STARTUP_CONCURRENCY_ENVELOPE.active_sizing_source
)
REQUEST_ADMISSION_CPU_MILLICORES = STARTUP_CONCURRENCY_ENVELOPE.cpu_millicores
REQUEST_ADMISSION_CPU_WEIGHT = STARTUP_CONCURRENCY_ENVELOPE.cpu_weight
REQUEST_ADMISSION_CPU_AFFINITY_COUNT = (
    STARTUP_CONCURRENCY_ENVELOPE.cpu_affinity_count
)
REQUEST_ADMISSION_CPU_SIZING_SOURCE = (
    STARTUP_CONCURRENCY_ENVELOPE.cpu_sizing_source
)
REQUEST_ADMISSION_CPU_ACTIVE_LIMIT = (
    STARTUP_CONCURRENCY_ENVELOPE.cpu_active_limit
)
REQUEST_ADMISSION_RESOURCE_ACTIVE_LIMIT = (
    STARTUP_CONCURRENCY_ENVELOPE.resource_active_limit
)
REQUEST_ADMISSION_STARTUP_MEMORY_AVAILABLE_BYTES = (
    STARTUP_CONCURRENCY_ENVELOPE.memory_available_bytes
)
REQUEST_ADMISSION_CONTROL_MEMORY_BUDGET_BYTES = (
    STARTUP_CONCURRENCY_ENVELOPE.memory_control_budget_bytes
)
REQUEST_ADMISSION_NOFILE_SOFT_LIMIT = (
    STARTUP_CONCURRENCY_ENVELOPE.nofile_soft_limit
)
REQUEST_ADMISSION_STARTUP_OPEN_FDS = STARTUP_CONCURRENCY_ENVELOPE.open_fds
REQUEST_ADMISSION_EPHEMERAL_PORT_COUNT = (
    STARTUP_CONCURRENCY_ENVELOPE.ephemeral_port_count
)
REQUEST_ADMISSION_EPHEMERAL_PORT_OCCUPANCY = (
    STARTUP_CONCURRENCY_ENVELOPE.ephemeral_port_occupancy
)
REQUEST_ADMISSION_EPHEMERAL_ACTIVE_LIMIT = (
    STARTUP_CONCURRENCY_ENVELOPE.ephemeral_active_limit
)
network_resource_governor = AdaptiveNetworkGovernor()
if RESOURCE_ADMISSION_MODE == "legacy":
    UVICORN_CONNECTION_LIMIT: int | None = _positive_env_int(
        "UVICORN_CONNECTION_LIMIT",
        STARTUP_CONCURRENCY_ENVELOPE.uvicorn_limit_concurrency,
    )
else:
    UVICORN_CONNECTION_LIMIT = None
# Uvicorn request-count limiting stays disabled. The custom protocol rejects
# only at live FD pressure and keeps the absolute header timeout.
UVICORN_LIMIT_CONCURRENCY = None
UVICORN_BACKLOG = _positive_env_int(
    "UVICORN_BACKLOG",
    kernel_somaxconn() or 2048,
)
UVICORN_HEADER_TIMEOUT_SECONDS = max(
    0.1,
    _env_float("UVICORN_HEADER_TIMEOUT_SECONDS", 5.0),
)
INBOUND_CONNECTION_ADMISSION = (
    network_resource_governor.allow_inbound_connection
)
UVICORN_HTTP_PROTOCOL, BOUNDED_HTTP_PROTOCOL_STATS = build_bounded_h11_protocol(
    connection_limit=UVICORN_CONNECTION_LIMIT,
    header_timeout_seconds=UVICORN_HEADER_TIMEOUT_SECONDS,
    connection_admission=INBOUND_CONNECTION_ADMISSION,
)
REQUEST_ADMISSION_WAIT_TIMEOUT_SECONDS = max(
    0.001,
    _env_float(
        "REQUEST_ADMISSION_WAIT_TIMEOUT_SECONDS",
        5.0,
    ),
)
REQUEST_CONTROL_MEMORY_RESERVATION_BYTES = _positive_env_int(
    "REQUEST_CONTROL_MEMORY_RESERVATION_BYTES",
    64 * 1024,
)
_startup_process_memory_capacity = process_memory_governor.maximum_capacity_bytes()
_startup_per_request_memory_limit = startup_per_request_memory_limit(
    process_memory_capacity_bytes=_startup_process_memory_capacity,
    # Per-request byte safety no longer depends on a process request count.
    active_limit=1,
)
_product_request_wire_body_max_bytes = _positive_env_int(
    "PRODUCT_REQUEST_MAX_BODY_BYTES",
    128 * 1024 * 1024,
)
_startup_large_request_memory_limit = startup_large_request_memory_limit(
    process_memory_capacity_bytes=_startup_process_memory_capacity,
    normal_request_limit_bytes=_startup_per_request_memory_limit,
    product_wire_limit_bytes=_product_request_wire_body_max_bytes,
    raw_memory_multiplier=DEFAULT_JSON_RAW_MEMORY_MULTIPLIER,
)
_large_request_process_ceiling = max(
    _startup_per_request_memory_limit,
    _startup_process_memory_capacity // 4,
)
REQUEST_BODY_RESERVATION_MAX_BYTES = _bounded_env_int(
    "REQUEST_BODY_RESERVATION_MAX_BYTES",
    _startup_large_request_memory_limit,
    _large_request_process_ceiling,
)
REQUEST_JSON_COMPLEXITY_MAX_BYTES = _bounded_env_int(
    "REQUEST_JSON_COMPLEXITY_MAX_BYTES",
    min(
        _startup_large_request_memory_limit,
        REQUEST_BODY_RESERVATION_MAX_BYTES,
    ),
    REQUEST_BODY_RESERVATION_MAX_BYTES,
)
_json_request_budget_multiplier = DEFAULT_JSON_RAW_MEMORY_MULTIPLIER + 1
_resource_safe_wire_body_max_bytes = max(
    1,
    min(
        _product_request_wire_body_max_bytes,
        REQUEST_JSON_COMPLEXITY_MAX_BYTES // _json_request_budget_multiplier,
        REQUEST_BODY_RESERVATION_MAX_BYTES // _json_request_budget_multiplier,
    ),
)
REQUEST_WIRE_BODY_MAX_BYTES = _bounded_env_int(
    "REQUEST_MAX_BODY_BYTES",
    _resource_safe_wire_body_max_bytes,
    _resource_safe_wire_body_max_bytes,
)
_request_wire_body_default = REQUEST_WIRE_BODY_MAX_BYTES
_legacy_zstd_wire_body_max_bytes = _bounded_env_int(
    "ZSTD_REQUEST_MAX_BODY_BYTES",
    _request_wire_body_default,
    _resource_safe_wire_body_max_bytes,
)
ZSTD_REQUEST_COMPRESSED_MAX_BYTES = _bounded_env_int(
    "ZSTD_REQUEST_MAX_COMPRESSED_BODY_BYTES",
    _legacy_zstd_wire_body_max_bytes,
    _resource_safe_wire_body_max_bytes,
)
ZSTD_REQUEST_DECOMPRESSED_MAX_BYTES = _bounded_env_int(
    "ZSTD_REQUEST_MAX_DECOMPRESSED_BODY_BYTES",
    _legacy_zstd_wire_body_max_bytes,
    _resource_safe_wire_body_max_bytes,
)
REQUEST_BODY_BUDGET_BYTES = max(
    1,
    _env_int(
        "REQUEST_BODY_BUDGET_BYTES",
        process_memory_governor.maximum_capacity_bytes(),
    ),
)
REQUEST_RESPONSE_RESERVATION_MAX_BYTES = max(
    1,
    _env_int(
        "REQUEST_RESPONSE_RESERVATION_MAX_BYTES",
        _startup_per_request_memory_limit,
    ),
)
REQUEST_LARGE_BODY_THRESHOLD_WEIGHTED_BYTES = _bounded_env_int(
    "REQUEST_LARGE_BODY_THRESHOLD_WEIGHTED_BYTES",
    min(
        _startup_per_request_memory_limit,
        _product_request_wire_body_max_bytes * 2,
    ),
    REQUEST_BODY_RESERVATION_MAX_BYTES,
)
_startup_large_body_limit_max = max(
    1,
    min(
        2,
        _startup_process_memory_capacity
        // max(1, REQUEST_BODY_RESERVATION_MAX_BYTES),
    ),
)
REQUEST_LARGE_BODY_LIMIT = _bounded_env_int(
    "REQUEST_LARGE_BODY_LIMIT",
    1,
    _startup_large_body_limit_max,
)
REQUEST_BODY_MEMORY_WAIT_TIMEOUT_SECONDS = max(
    0.001,
    _env_float(
        "REQUEST_BODY_MEMORY_WAIT_TIMEOUT_SECONDS",
        30.0 if RESOURCE_ADMISSION_MODE == "weighted" else 1.0,
    ),
)
REQUEST_BODY_MEMORY_WAITER_LIMIT: int | None = (
    None
    if RESOURCE_ADMISSION_MODE == "weighted"
    else max(
        1,
        _env_int(
            "REQUEST_BODY_MEMORY_WAITER_LIMIT",
            REQUEST_ADMISSION_ACTIVE_LIMIT,
        ),
    )
)
REQUEST_SMALL_BODY_LANE_THRESHOLD_BYTES = _bounded_env_int(
    "REQUEST_SMALL_BODY_LANE_THRESHOLD_BYTES",
    min(4 * 1024 * 1024, REQUEST_BODY_RESERVATION_MAX_BYTES),
    REQUEST_BODY_RESERVATION_MAX_BYTES,
)
REQUEST_SMALL_BODY_LANE_RESERVE_BYTES = _bounded_env_int(
    "REQUEST_SMALL_BODY_LANE_RESERVE_BYTES",
    min(64 * 1024 * 1024, max(1, REQUEST_BODY_BUDGET_BYTES // 8)),
    REQUEST_BODY_BUDGET_BYTES,
)
UPSTREAM_POOL_SIZE: int | None = (
    None
    if RESOURCE_ADMISSION_MODE == "weighted"
    else max(
        1,
        _env_int("UPSTREAM_POOL_SIZE", REQUEST_ADMISSION_CPU_ACTIVE_LIMIT),
    )
)
UPSTREAM_POOL_WAITER_LIMIT: int | None = (
    None
    if RESOURCE_ADMISSION_MODE == "weighted"
    else max(
        0,
        _env_int(
            "UPSTREAM_POOL_WAITER_LIMIT",
            REQUEST_ADMISSION_ACTIVE_LIMIT,
        ),
    )
)
UPSTREAM_POOL_WAIT_TIMEOUT_SECONDS = max(
    0.001,
    _env_float("UPSTREAM_POOL_WAIT_TIMEOUT_SECONDS", 5.0),
)
UPSTREAM_HTTPX_POOL_TIMEOUT_SECONDS = max(
    0.001,
    _env_float("UPSTREAM_HTTPX_POOL_TIMEOUT_SECONDS", 1.0),
)

request_admission_controller = RequestAdmissionController(
    capacity=(
        REQUEST_ADMISSION_ACTIVE_LIMIT
        if RESOURCE_ADMISSION_MODE == "legacy"
        else None
    ),
    waiter_limit=(
        REQUEST_ADMISSION_WAITER_LIMIT
        if RESOURCE_ADMISSION_MODE == "legacy"
        else None
    ),
    wait_timeout_seconds=REQUEST_ADMISSION_WAIT_TIMEOUT_SECONDS,
    control_memory_bytes=REQUEST_CONTROL_MEMORY_RESERVATION_BYTES,
    max_body_bytes=REQUEST_BODY_RESERVATION_MAX_BYTES,
    body_budget_bytes=REQUEST_BODY_BUDGET_BYTES,
    max_response_bytes=REQUEST_RESPONSE_RESERVATION_MAX_BYTES,
    large_body_threshold_weighted_bytes=(
        REQUEST_LARGE_BODY_THRESHOLD_WEIGHTED_BYTES
        if RESOURCE_ADMISSION_MODE == "legacy"
        else 0
    ),
    large_body_limit=(
        REQUEST_LARGE_BODY_LIMIT
        if RESOURCE_ADMISSION_MODE == "legacy"
        else 0
    ),
    body_memory_wait_timeout_seconds=(
        REQUEST_BODY_MEMORY_WAIT_TIMEOUT_SECONDS
    ),
    body_memory_waiter_limit=REQUEST_BODY_MEMORY_WAITER_LIMIT,
    small_body_lane_threshold_bytes=REQUEST_SMALL_BODY_LANE_THRESHOLD_BYTES,
    small_body_lane_reserve_bytes=REQUEST_SMALL_BODY_LANE_RESERVE_BYTES,
    memory_governor=process_memory_governor,
    decision_observer=emit_large_body_admission_decision,
    response_buffer_observer=emit_response_buffer_event,
)
idempotency_coordinator = build_default_idempotency_coordinator(
    memory_governor=process_memory_governor
)
runtime_gauges.attach_request_admission(request_admission_controller.snapshot)
runtime_gauges.attach_stream_parser_budget(stream_parser_retained_budget_snapshot)


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


def _socket_inode_for_fd(fd: int) -> Optional[str]:
    try:
        target = os.readlink(f"/proc/self/fd/{int(fd)}")
    except (OSError, TypeError, ValueError):
        return None
    match = re.match(r"^socket:\[(\d+)\]$", target)
    return match.group(1) if match else None


def _httpcore_connection_socket_inode(connection: Any) -> Optional[str]:
    current = connection
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        network_stream = getattr(current, "_network_stream", None)
        get_extra_info = getattr(network_stream, "get_extra_info", None)
        if callable(get_extra_info):
            try:
                sock = get_extra_info("socket")
            except BaseException:
                sock = None
            fileno = getattr(sock, "fileno", None)
            if callable(fileno):
                try:
                    inode = _socket_inode_for_fd(fileno())
                except BaseException:
                    inode = None
                if inode is not None:
                    return inode
        next_connection = getattr(current, "_connection", None)
        if next_connection is None or next_connection is current:
            break
        current = next_connection
    return None


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


async def _force_close_httpcore_stream_chain_observed_safely(
    upstream_response: Any,
    outcome_sink: Callable[[dict[str, Any]], None],
) -> bool:
    return await force_close_response_httpcore_stream_chain_safely(
        upstream_response,
        label="Upstream HTTP response stream",
        outcome_sink=outcome_sink,
    )


async def _close_upstream_response_safely(upstream_response: Any | None) -> bool:
    if upstream_response is None:
        return True
    # This helper owns the complete two-stage close: a short cooperative
    # response close, followed by pool eviction/transport abort before any
    # fail-closed wait.  Calling a generic aclose helper first would prevent
    # the eviction phase from ever running when aclose ignores cancellation.
    return await _force_close_httpcore_stream_chain_safely(upstream_response)


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


async def _close_observed_responses_upstream_stream_safely(
    stream_cm: Any | None,
    upstream_response: Any | None,
    diagnostics: ResponsesStreamDiagnostics,
    *,
    owner: str,
    trigger: str,
) -> bool:
    claimed = diagnostics.begin_cleanup(owner=owner, trigger=trigger)
    cleanup_outcome: dict[str, Any] = {}
    transport_safe = await _force_close_httpcore_stream_chain_observed_safely(
        upstream_response,
        cleanup_outcome.update,
    ) if upstream_response is not None else True
    context_exit_succeeded = await _close_stream_cm_safely(stream_cm)
    cleanup_outcome["won_cleanup_claim"] = claimed
    diagnostics.record_cleanup_action(
        actor=owner,
        trigger=trigger,
        outcome=cleanup_outcome,
        transport_safe=transport_safe,
        context_exit_succeeded=context_exit_succeeded,
    )
    diagnostics.observe_cleanup_transport_outcome(
        cleanup_outcome,
        actor=owner,
    )
    diagnostics.finish_cleanup(
        transport_safe=transport_safe,
        context_exit_succeeded=context_exit_succeeded,
        actor=owner,
    )
    return bool(transport_safe and context_exit_succeeded)


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
    sweeper_trackers: list[ResponsesStreamDiagnostics] = []
    def collect_connection(connection: Any) -> None:
        if all(candidate is not connection for candidate in closing):
            closing.append(connection)

    def close_reason(connection: Any) -> tuple[str | None, Optional[str]]:
        inode = _httpcore_connection_socket_inode(connection)
        # Do not treat kernel CLOSE_WAIT as a close reason.  A peer FIN may be
        # visible while an active response still has unread bytes queued ahead
        # of it.  Only httpcore's state machine can establish that the
        # connection is closed or expired without truncating that response.
        is_closed = getattr(connection, "is_closed", None)
        has_expired = getattr(connection, "has_expired", None)
        if callable(is_closed) and is_closed():
            return "httpcore_is_closed", inode
        if callable(has_expired) and has_expired():
            return "httpcore_has_expired", inode
        return None, inode

    def record_sweeper_close(connection: Any, trigger: str, inode: Optional[str]) -> None:
        if inode is None:
            inode = _httpcore_connection_socket_inode(connection)
        observed_trackers = observe_pool_sweeper_connection_close(
            connection,
            trigger=trigger,
            socket_inode=inode,
        )
        for tracker in observed_trackers:
            if all(existing is not tracker for existing in sweeper_trackers):
                sweeper_trackers.append(tracker)

    if lock is not None:
        with lock:
            for connection in list(pool_connections):
                trigger, inode = close_reason(connection)
                if trigger is not None:
                    record_sweeper_close(connection, trigger, inode)
                    if connection in pool_connections:
                        pool_connections.remove(connection)
                    collect_connection(connection)
            for connection in assign_requests():
                record_sweeper_close(
                    connection,
                    "httpcore_assign_cleanup",
                    _httpcore_connection_socket_inode(connection),
                )
                collect_connection(connection)
    else:
        for connection in list(pool_connections):
            trigger, inode = close_reason(connection)
            if trigger is not None:
                record_sweeper_close(connection, trigger, inode)
                if connection in pool_connections:
                    pool_connections.remove(connection)
                collect_connection(connection)
        for connection in assign_requests():
            record_sweeper_close(
                connection,
                "httpcore_assign_cleanup",
                _httpcore_connection_socket_inode(connection),
            )
            collect_connection(connection)
    if not closing:
        return 0
    try:
        await close_connections(closing)
    except BaseException:
        for tracker in sweeper_trackers:
            tracker.observe_pool_sweeper_close_completed(succeeded=False)
        raise
    for tracker in sweeper_trackers:
        tracker.observe_pool_sweeper_close_completed(succeeded=True)
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
    matching_provider_count = max(0, int(plan.num_matching_providers or 0))
    current_info["matching_provider_count"] = matching_provider_count
    current_info["planned_attempt_count"] = (
        matching_provider_count + current_info["planned_retry_count"]
    )
    current_info.setdefault("attempt_count", 0)
    current_info.setdefault("retry_decision_count", 0)
    current_info.setdefault("retry_transition_count", 0)


def _record_retry_observability(attempt: Any, status_code: int, error_message: Any) -> None:
    info = get_request_info()
    if not isinstance(info, dict):
        return
    retry_count = int(info.get("retry_count") or 0) + 1
    info["retry_count"] = retry_count
    # Preserve the legacy counter while exposing its exact meaning.  The
    # runner owns this counter too, so max() keeps callbacks idempotent.
    info["retry_decision_count"] = max(
        retry_count,
        int(info.get("retry_decision_count") or 0),
    )
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
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
        VERSION = data['project']['version']
except Exception:
    VERSION = 'unknown'
logger.info("VERSION: %s", VERSION)

PUBLIC_HEALTH_PATHS = {"/healthz"}
rust_responses_control_plane = RustResponsesControlPlane()


def _is_public_health_request(request: Request) -> bool:
    return request.method in {"GET", "HEAD"} and request.url.path in PUBLIC_HEALTH_PATHS


def _has_valid_rust_control_token(request: Request) -> bool:
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        return _has_valid_rust_control_scope(scope)
    if not RUST_RESPONSES_CONTROL_TOKEN:
        return False
    headers = getattr(request, "headers", None)
    supplied = str(
        headers.get(RUST_RESPONSES_CONTROL_HEADER) or ""
        if hasattr(headers, "get")
        else ""
    )
    return bool(supplied) and secrets.compare_digest(
        supplied,
        RUST_RESPONSES_CONTROL_TOKEN,
    )


def _scope_header_values(scope: dict[str, Any], header: str) -> list[str]:
    wanted = header.encode("ascii")
    return [
        value.decode("latin-1", errors="replace")
        for name, value in (scope.get("headers") or [])
        if name.lower() == wanted
    ]


def _has_valid_rust_control_scope(scope: dict[str, Any]) -> bool:
    if not RUST_RESPONSES_CONTROL_TOKEN:
        return False
    supplied_values = _scope_header_values(
        scope,
        RUST_RESPONSES_CONTROL_HEADER,
    )
    return len(supplied_values) == 1 and secrets.compare_digest(
        supplied_values[0],
        RUST_RESPONSES_CONTROL_TOKEN,
    )


def _is_trusted_rust_spooled_request(scope: dict[str, Any]) -> bool:
    if (
        str(scope.get("method") or "").upper() != "POST"
        or str(scope.get("path") or "") != "/v1/responses"
        or not _has_valid_rust_control_scope(scope)
    ):
        return False
    tiers = _scope_header_values(scope, RUST_REQUEST_SPOOL_TIER_HEADER)
    body_bytes = _scope_header_values(
        scope,
        "x-uni-api-rust-request-spool-body-bytes",
    )
    disk_bytes = _scope_header_values(
        scope,
        "x-uni-api-rust-request-spool-local-disk-bytes",
    )
    if len(tiers) != 1 or tiers[0] != "local_disk":
        return False
    if len(body_bytes) != 1 or len(disk_bytes) != 1:
        return False
    try:
        observed_body_bytes = int(body_bytes[0])
        observed_disk_bytes = int(disk_bytes[0])
    except ValueError:
        return False
    return (
        observed_body_bytes >= 0
        and observed_body_bytes == observed_disk_bytes
    )


def _is_internal_rust_control_request(request: Request) -> bool:
    return bool(
        request.url.path.startswith(RUST_RESPONSES_INTERNAL_PREFIX)
        and _has_valid_rust_control_token(request)
    )


def _record_rust_request_spool_observability(
    request: Request,
    current_info: dict[str, Any],
) -> None:
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        _record_rust_request_spool_observability_from_scope(
            scope,
            current_info,
        )
        return
    if not _has_valid_rust_control_token(request):
        return
    observed = False
    for header, field in RUST_REQUEST_SPOOL_INT_HEADERS.items():
        raw = request.headers.get(header)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        current_info[field] = value
        observed = True
    tier = str(request.headers.get(RUST_REQUEST_SPOOL_TIER_HEADER) or "").strip()
    if tier == "local_disk":
        current_info["rust_request_spool_final_tier"] = tier
        observed = True
    if observed:
        current_info["rust_request_spool"] = True


def _record_rust_request_spool_observability_from_scope(
    scope: dict[str, Any],
    current_info: dict[str, Any],
) -> None:
    if not _is_trusted_rust_spooled_request(scope):
        return
    for header, field in RUST_REQUEST_SPOOL_INT_HEADERS.items():
        values = _scope_header_values(scope, header)
        if len(values) != 1:
            continue
        try:
            value = int(values[0])
        except ValueError:
            continue
        if value >= 0:
            current_info[field] = value
    current_info["rust_request_spool_final_tier"] = "local_disk"
    current_info["rust_request_spool"] = True


def _is_middleware_bypass_request(request: Request) -> bool:
    return _is_public_health_request(request) or _is_internal_rust_control_request(
        request
    )


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

    try:
        app.state.rust_responses_snapshot_revision = (
            publish_rust_responses_snapshot(
                config,
                list(api_list),
                database_disabled=DISABLE_DATABASE,
            )
        )
    except Exception:
        # A missing snapshot only disables the native fast path. It must never
        # make Python configuration refresh or serving unavailable.
        logger.exception("Failed to publish Rust Responses config snapshot")

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
    logger.info(
        "Logical idempotency coordinator: %s",
        json.dumps(
            idempotency_coordinator.snapshot(),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
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
        app.state.client_manager = ClientManager(
            pool_size=UPSTREAM_POOL_SIZE,
            waiter_limit=UPSTREAM_POOL_WAITER_LIMIT,
            wait_timeout_seconds=UPSTREAM_POOL_WAIT_TIMEOUT_SECONDS,
            pool_timeout_seconds=UPSTREAM_HTTPX_POOL_TIMEOUT_SECONDS,
            network_governor=network_resource_governor,
        )
        await app.state.client_manager.init(default_config)
        runtime_gauges.attach_upstream_client(app.state.client_manager.snapshot)

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

    await runtime_gauges.start_network_sampler()
    await _start_responses_stream_stats_workers()
    await start_fugue_observability_from_env(service_version=VERSION)
    try:
        await worker_runtime_observer.start()
    except Exception:
        logger.exception("Worker runtime observability failed to start")

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
    await _stop_responses_stream_stats_workers(timeout=5.0)
    try:
        await worker_runtime_observer.stop()
    except Exception:
        logger.exception("Worker runtime observability failed to stop cleanly")
    await stop_fugue_observability()
    await runtime_gauges.stop_network_sampler()
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
    if request.method == "POST" and is_json_media_type(
        request.headers.get("content-type", "")
    ):
        try:
            body_bytes = await request.body()
            request.state.uni_api_request_body_bytes = len(body_bytes)
            if not body_bytes:
                return None
            return await run_json_cpu(json.loads, body_bytes)
        except json.JSONDecodeError:
            return None
    return None


def _request_content_length_bytes(http_request: Optional[Request]) -> int:
    if http_request is None:
        return 0
    headers = getattr(http_request, "headers", None)
    raw = headers.get("content-length") if headers is not None else None
    if not raw:
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _request_body_size_bytes(http_request: Optional[Request], body: Any = None) -> int:
    if http_request is not None:
        state = getattr(http_request, "state", None)
        state_bytes = getattr(state, "uni_api_request_body_bytes", None)
        if isinstance(state_bytes, int) and state_bytes > 0:
            return state_bytes

    content_length = _request_content_length_bytes(http_request)
    if content_length > 0:
        return content_length
    if body is None:
        return 0

    try:
        if isinstance(body, BaseModel):
            payload = body.model_dump(mode="json")
        else:
            payload = jsonable_encoder(body)
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
    except Exception:
        return 0


async def _drop_parsed_request_caches(http_request: Request) -> None:
    """Drop FastAPI's raw/JSON cache after endpoint model validation."""

    raw_body = getattr(http_request, "_body", None)
    raw_bytes = (
        len(raw_body)
        if isinstance(raw_body, (bytes, bytearray, memoryview))
        else 0
    )
    if raw_bytes:
        http_request.state.uni_api_request_body_bytes = raw_bytes
    if hasattr(http_request, "_body"):
        http_request._body = b""
    if hasattr(http_request, "_json"):
        delattr(http_request, "_json")
    raw_body = None
    lease = get_request_admission_lease()
    if lease is not None and raw_bytes:
        await lease.release_body_bytes(
            min(raw_bytes, max(0, int(lease.reserved_body_bytes)))
        )


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
    def __init__(
        self,
        cooldown_period=300,
        *,
        route_circuit: ProviderModelCircuitBreaker | None = None,
    ):
        self._excluded_models = defaultdict(lambda: None)
        self.cooldown_period = cooldown_period
        self.route_circuit = (
            route_circuit or ProviderModelCircuitBreaker.from_environment()
        )
        self.has_route_circuit = True

    async def exclude_model(self, provider: str, model: str):
        model_key = f"{provider}/{model}"
        self._excluded_models[model_key] = datetime.now()

    def _is_legacy_model_excluded(
        self,
        provider: str,
        model: str,
        cooldown_period: int,
    ) -> bool:
        model_key = f"{provider}/{model}"
        excluded_time = self._excluded_models.get(model_key)
        if not excluded_time:
            return False
        if datetime.now() - excluded_time > timedelta(seconds=cooldown_period):
            del self._excluded_models[model_key]
            return False
        return True

    async def is_model_excluded(self, provider: str, model: str, cooldown_period=0) -> bool:
        legacy_excluded = self._is_legacy_model_excluded(
            provider,
            model,
            cooldown_period,
        )
        return legacy_excluded or self.route_circuit.is_open(provider, model)

    async def record_model_failure(
        self,
        provider: str,
        model: str,
        status_code: int,
    ) -> bool:
        return self.route_circuit.record_failure(provider, model, status_code)

    async def record_model_success(self, provider: str, model: str) -> bool:
        return self.route_circuit.record_success(provider, model)

    def snapshot(self) -> dict[str, Any]:
        return {
            "legacy_cooldown_period_seconds": self.cooldown_period,
            "legacy_excluded_model_count": sum(
                1 for value in self._excluded_models.values() if value
            ),
            "provider_model_circuit": self.route_circuit.snapshot(),
        }

    async def get_available_providers(self, providers: list) -> list:
        """过滤出可用的providers，仅排除不可用的模型"""
        available_providers = []
        apply_legacy_cooldown = len(providers) > 1
        for provider in providers:
            provider_name = provider['provider']
            model_dict = provider['model'][0]  # 获取唯一的模型字典
            # source_model = list(model_dict.keys())[0]  # 源模型名称
            target_model = list(model_dict.values())[0]  # 目标模型名称
            cooldown_period = provider.get('preferences', {}).get('cooldown_period', self.cooldown_period)

            # 检查该模型是否被排除
            circuit_open = self.route_circuit.is_open(
                provider_name,
                target_model,
            )
            legacy_excluded = bool(
                apply_legacy_cooldown
                and self._is_legacy_model_excluded(
                    provider_name,
                    target_model,
                    cooldown_period,
                )
            )
            if not circuit_open and not legacy_excluded:
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
        return True

    # 在成功请求时，快照当前价格，写入数据库
    try:
        if current_info.get("success") and current_info.get("model"):
            prompt_price, completion_price = get_current_model_prices(current_info["model"])
            current_info["prompt_price"] = prompt_price
            current_info["completion_price"] = completion_price
    except Exception:
        pass

    try:
        persisted = await stats_repository.add_request_stat(current_info)
        if persisted is False:
            return False
        check_key = current_info["api_key"]
        if check_key and check_key in app.state.paid_api_keys_states and current_info["total_tokens"] > 0:
            await update_paid_api_keys_states(app, check_key)
        return True
    except Exception as e:
        logger.error(f"Error acquiring database lock: {str(e)}")
        if is_debug:
            import traceback
            traceback.print_exc()
        return False

async def update_channel_stats(request_id, provider, model, api_key, success, provider_api_key: str = None):
    if DISABLE_DATABASE:
        return True

    try:
        persisted = await stats_repository.add_channel_stat(
            request_id=request_id,
            provider=provider,
            model=model,
            api_key=api_key,
            provider_api_key=provider_api_key,
            success=success,
        )
        return persisted is not False
    except Exception as e:
        logger.error(f"Error acquiring database lock: {str(e)}")
        if is_debug:
            import traceback
            traceback.print_exc()
        return False

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
    except Exception as exc:
        # Unknown receive errors are not equivalent to a peer disconnect.
        logger.warning("request disconnect monitor stopped: %s", exc)


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
        is_public_health_request=_is_middleware_bypass_request,
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
        responses_usage_buffer_limit_bytes=RESPONSES_CANONICAL_EVENT_MAX_BYTES,
        logging_response_class=LoggingStreamingResponse,
        debug=is_debug,
    ),
)

app.add_middleware(
    RequestBodyDecompressionMiddleware,
    max_identity_body_bytes=REQUEST_WIRE_BODY_MAX_BYTES,
    max_zstd_compressed_body_bytes=ZSTD_REQUEST_COMPRESSED_MAX_BYTES,
    max_zstd_decompressed_body_bytes=ZSTD_REQUEST_DECOMPRESSED_MAX_BYTES,
    json_max_estimated_bytes=REQUEST_JSON_COMPLEXITY_MAX_BYTES,
    resource_only_body_admission=_is_trusted_rust_spooled_request,
)

@app.middleware("http")
async def ensure_config(request: Request, call_next):
    if _is_middleware_bypass_request(request):
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


def _observe_idempotency_claim(
    event: str,
    fields: dict[str, Any],
) -> None:
    if event not in {"wait", "replay", "conflict", "unavailable"}:
        return
    trace_logger.info(
        "idempotency event=%s key_fingerprint=%s method=%s path=%s",
        event,
        fields.get("key_fingerprint"),
        fields.get("method"),
        fields.get("path"),
    )


app.add_middleware(
    IdempotencyMiddleware,
    coordinator=idempotency_coordinator,
    enabled=_env_bool("IDEMPOTENCY_ENABLED", True),
    max_request_body_bytes=max(
        REQUEST_WIRE_BODY_MAX_BYTES,
        ZSTD_REQUEST_COMPRESSED_MAX_BYTES,
    ),
    request_body_idle_timeout_seconds=max(
        0.001,
        _env_float("REQUEST_BODY_IDLE_TIMEOUT_SECONDS", 15.0),
    ),
    request_body_total_timeout_seconds=max(
        0.001,
        _env_float("REQUEST_BODY_TOTAL_TIMEOUT_SECONDS", 120.0),
    ),
    wait_timeout_seconds=max(
        0.001,
        _env_float("IDEMPOTENCY_WAIT_TIMEOUT_SECONDS", 30 * 60),
    ),
    observer=_observe_idempotency_claim,
    phase_sample_decider=lambda: (
        worker_runtime_observer.should_sample_phase_request("idempotency_hash")
    ),
    phase_observer=worker_runtime_observer.record_phase_sample,
)


def _bypass_request_admission(scope: dict[str, Any]) -> bool:
    path = str(scope.get("path") or "")
    if path.startswith(
        RUST_RESPONSES_INTERNAL_PREFIX
    ) and _has_valid_rust_control_scope(scope):
        return True
    return (
        str(scope.get("method") or "").upper() in {"GET", "HEAD"}
        and path in {*PUBLIC_HEALTH_PATHS, "/v1/observability/runtime"}
    )


def _observe_request_admission_rejection(
    scope: dict[str, Any],
    rejection: Any,
    wait_ms: float,
) -> None:
    _observe_early_request_outcome(
        scope,
        status_code=int(rejection.status_code),
        reason=str(rejection.reason),
        wait_ms=wait_ms,
        admission_rejected=True,
    )


def _observe_request_admission_response_write(
    scope: dict[str, Any],
    rejection: Any,
    completed: bool,
) -> None:
    if int(getattr(rejection, "status_code", 0) or 0) != 503:
        return
    completed_total, failed_total = runtime_gauges.record_admission_503_response_write(
        str(getattr(rejection, "reason", "unknown") or "unknown"),
        completed=bool(completed),
    )
    state = scope.get("state")
    request_id = "untracked"
    trace_id = "untracked"
    lease_id = None
    if isinstance(state, dict):
        request_id = str(
            state.get(ADMISSION_REQUEST_ID_STATE_KEY) or request_id
        )
        trace_id = str(state.get(ADMISSION_TRACE_ID_STATE_KEY) or trace_id)
        lease = state.get(ADMISSION_LEASE_STATE_KEY)
        lease_id = str(getattr(lease, "lease_id", "") or "") or None
    emit_admission_503_response_write_outcome(
        Admission503ResponseWriteOutcome(
            schema_version=1,
            occurred_at_unix_ms=int(time() * 1000.0),
            reason=str(getattr(rejection, "reason", "unknown") or "unknown"),
            intended_status_code=503,
            asgi_response_write_completed=bool(completed),
            request_self_lease_id=lease_id,
            request_self_request_id=request_id,
            request_self_trace_id=trace_id,
            request_self_method=str(scope.get("method") or "")[:16] or None,
            request_self_path=str(scope.get("path") or "")[:160] or None,
            runtime_global_admission_503_response_write_completed_total_after=(
                completed_total
            ),
            runtime_global_admission_503_response_write_failed_total_after=(
                failed_total
            ),
        )
    )


def _observe_early_body_response(
    scope: dict[str, Any],
    status_code: int,
    reason: str,
) -> None:
    state = scope.get("state")
    wait_ms = 0.0
    if isinstance(state, dict):
        try:
            wait_ms = float(state.get("uni_api_admission_wait_ms") or 0.0)
        except (TypeError, ValueError):
            wait_ms = 0.0
    _observe_early_request_outcome(
        scope,
        status_code=status_code,
        reason=reason,
        wait_ms=max(0.0, wait_ms),
        admission_rejected=status_code == 413,
    )


def _observe_early_request_outcome(
    scope: dict[str, Any],
    *,
    status_code: int,
    reason: str,
    wait_ms: float,
    admission_rejected: bool,
) -> None:
    downstream_disconnected = int(status_code) == 499 and str(reason) in {
        "request_body_disconnected",
        "admission_wait_disconnected",
        "disconnected_while_queued",
        "request_disconnected",
    }
    state = scope.get("state")
    body_complexity_diagnostics = (
        request_body_complexity_diagnostics_from_scope(scope)
    )
    if isinstance(state, dict):
        existing_info = state.get("uni_api_request_info")
        if isinstance(existing_info, dict):
            _record_rust_request_spool_observability_from_scope(
                scope,
                existing_info,
            )
            existing_info["status_code"] = int(status_code)
            existing_info["error_type"] = str(reason)
            existing_info["success"] = False
            if admission_rejected:
                existing_info["admission_rejected"] = True
                existing_info["admission_reason"] = str(reason)
            if (
                int(status_code) in {503, 507}
                and existing_info.get("rust_request_spool")
            ):
                existing_info["rust_request_spool_failure_resource"] = str(
                    reason
                )
            if downstream_disconnected:
                existing_info["downstream_disconnected"] = True
                existing_info["stream_outcome"] = "downstream_disconnected"
            if body_complexity_diagnostics:
                existing_info.setdefault(
                    REQUEST_BODY_COMPLEXITY_INFO_KEY,
                    body_complexity_diagnostics,
                )
            # Stats owns normal emission once it has created request context.
            if not existing_info.get("_fugue_observability_emitted"):
                _emit_request_observability(existing_info)
            return

    headers = {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in (scope.get("headers") or [])
    }
    incoming = _incoming_trace_context(headers)
    admission_request_id = ""
    admission_trace_id = ""
    if isinstance(state, dict):
        admission_request_id = str(
            state.get(ADMISSION_REQUEST_ID_STATE_KEY) or ""
        ).strip()
        admission_trace_id = str(
            state.get(ADMISSION_TRACE_ID_STATE_KEY) or ""
        ).strip().lower()
    if _is_valid_w3c_trace_id(admission_trace_id):
        incoming["trace_id"] = admission_trace_id
    trace = RequestTrace(
        trace_id=incoming["trace_id"],
        parent_span_id=incoming.get("parent_span_id"),
        trace_flags=incoming.get("trace_flags"),
        tracestate=incoming.get("tracestate"),
    )
    trace.mark("request_received")
    trace.add_ms("request_admission_wait_ms", wait_ms)
    if body_complexity_diagnostics:
        trace.mark("request_body_rejected")
    method = str(scope.get("method") or "").upper()
    path = str(scope.get("path") or "/")
    started_at = time() - max(0.0, wait_ms) / 1000.0
    current_info = {
        "trace_id": trace.trace_id,
        "parent_span_id": trace.parent_span_id,
        "request_id": (
            _normalize_request_id(admission_request_id)
            if admission_request_id
            else incoming.get("x_request_id") or trace.trace_id
        ),
        "endpoint": f"{method} {path}".strip(),
        "request_kind": path,
        "stream": False,
        "status_code": int(status_code),
        "error_type": str(reason),
        "success": False,
        "start_time": started_at,
        "process_time": max(0.0, time() - started_at),
        "trace": trace,
        "timing_spans": trace.snapshot(),
    }
    _record_rust_request_spool_observability_from_scope(scope, current_info)
    if admission_rejected:
        current_info["admission_rejected"] = True
        current_info["admission_reason"] = str(reason)
    if int(status_code) in {503, 507} and current_info.get(
        "rust_request_spool"
    ):
        current_info["rust_request_spool_failure_resource"] = str(reason)
    if body_complexity_diagnostics:
        current_info[REQUEST_BODY_COMPLEXITY_INFO_KEY] = (
            body_complexity_diagnostics
        )
    if downstream_disconnected:
        current_info["downstream_disconnected"] = True
        current_info["stream_outcome"] = "downstream_disconnected"
    _emit_request_observability(current_info)


# This must remain the last-added middleware so it is the outermost ASGI
# boundary and owns the lease through the complete streaming response.
app.add_middleware(
    RequestAdmissionMiddleware,
    controller=request_admission_controller,
    bypass=_bypass_request_admission,
    resource_only_body_admission=_is_trusted_rust_spooled_request,
    on_rejection=_observe_request_admission_rejection,
    on_early_response=_observe_early_body_response,
    on_rejection_response_write=_observe_request_admission_response_write,
)


def _require_rust_control_request(request: Request) -> None:
    if not _is_internal_rust_control_request(request):
        raise HTTPException(status_code=404, detail="Not found")


@app.get(
    f"{RUST_RESPONSES_INTERNAL_PREFIX}/{{session_id}}/plan",
    include_in_schema=False,
)
async def rust_responses_plan(request: Request, session_id: str):
    _require_rust_control_request(request)
    try:
        session = rust_responses_control_plane.get(session_id)
        return JSONResponse(content=await session.next_message())
    except RustResponsesControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    f"{RUST_RESPONSES_INTERNAL_PREFIX}/{{session_id}}/advance",
    include_in_schema=False,
)
async def rust_responses_advance(
    request: Request,
    session_id: str,
    outcome: dict[str, Any] = Body(...),
):
    _require_rust_control_request(request)
    try:
        session = rust_responses_control_plane.get(session_id)
        message = await session.advance(
            attempt_id=str(outcome.get("attempt_id") or ""),
            outcome=outcome,
        )
        return JSONResponse(content=message)
    except RustResponsesControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    f"{RUST_RESPONSES_INTERNAL_PREFIX}/{{session_id}}/commit",
    include_in_schema=False,
)
async def rust_responses_commit(
    request: Request,
    session_id: str,
    observation: dict[str, Any] = Body(...),
):
    _require_rust_control_request(request)
    try:
        session = rust_responses_control_plane.get(session_id)
        message = await session.observe_commit(
            attempt_id=str(observation.get("attempt_id") or ""),
            observation=observation,
        )
        return JSONResponse(content=message)
    except RustResponsesControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    f"{RUST_RESPONSES_INTERNAL_PREFIX}/{{session_id}}/complete",
    include_in_schema=False,
)
async def rust_responses_complete(
    request: Request,
    session_id: str,
    outcome: dict[str, Any] = Body(...),
):
    _require_rust_control_request(request)
    try:
        session = rust_responses_control_plane.get(session_id)
        message = await session.complete(
            attempt_id=str(outcome.get("attempt_id") or ""),
            outcome=outcome,
        )
        return JSONResponse(content=message)
    except RustResponsesControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return await healthz_response(VERSION)


@app.get("/v1/observability/runtime", include_in_schema=False)
async def observability_runtime():
    return await observability_runtime_response(
        runtime_gauges,
        getattr(app.state, "client_manager", None),
        channel_manager=getattr(app.state, "channel_manager", None),
        stream_cleanup_snapshot=background_stream_cleanup_snapshot,
        provider_key_pools_snapshot=lambda: provider_key_pools_snapshot(app),
        idempotency_snapshot=idempotency_coordinator.snapshot,
    )


class ClientManager(ClientPool):
    def __init__(
        self,
        pool_size=UPSTREAM_POOL_SIZE,
        *,
        waiter_limit=UPSTREAM_POOL_WAITER_LIMIT,
        wait_timeout_seconds=UPSTREAM_POOL_WAIT_TIMEOUT_SECONDS,
        pool_timeout_seconds=UPSTREAM_HTTPX_POOL_TIMEOUT_SECONDS,
        network_governor=network_resource_governor,
    ):
        super().__init__(
            pool_size=pool_size,
            waiter_limit=waiter_limit,
            wait_timeout_seconds=wait_timeout_seconds,
            pool_timeout_seconds=pool_timeout_seconds,
            sweep_client=_sweep_httpx_client_idle_connections,
            current_trace=_current_trace,
            begin_upstream_pool=runtime_gauges.begin_upstream_pool,
            end_upstream_pool=runtime_gauges.end_upstream_pool,
            record_upstream_wait=runtime_gauges.record_upstream_pool_wait,
            network_governor=network_governor,
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


def _postcommit_sse_protocol_error_event() -> bytes:
    """Return a protocol-valid, payload-redacted synthetic SSE terminal."""

    return (
        "event: error\n"
        "data: "
        + json.dumps(
            {
                "type": "error",
                "error": {
                    "message": "Upstream SSE protocol error",
                    "type": "stream_error",
                    "code": "upstream_sse_protocol_error",
                    "status_code": 502,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\n"
    ).encode("utf-8")


def _record_postcommit_sse_protocol_error_isolation(
    current_info: dict[str, Any],
    exc: SSEProtocolError,
) -> None:
    current_info["success"] = False
    current_info["stream_outcome"] = "upstream_stream_abort"
    current_info["stream_error_status_code"] = 502
    current_info["stream_error_event_type"] = "error"
    current_info["stream_error_after_response_start"] = True
    current_info["error_type"] = type(exc).__name__
    current_info["postcommit_sse_protocol_error_isolated"] = True
    image_diagnostics = current_info.get("image_stream_diagnostics")
    if isinstance(image_diagnostics, dict):
        image_diagnostics["synthetic_terminal"] = True
        image_diagnostics["synthetic_terminal_type"] = "error"


async def _track_legacy_stream_outcome(
    source: Any,
    *,
    current_info: dict[str, Any],
    channel_id: str,
    model: str,
    provider_api_key: Optional[str],
    fallback_background_tasks: BackgroundTasks,
    response_memory_lease: Any = None,
):
    """Finalize legacy channel accounting when the stream really terminates.

    Returning a ``StreamingResponse`` only transfers ownership of the upstream
    iterator; it does not prove that the provider completed successfully.  The
    channel result therefore belongs to this iterator's terminal outcome, not
    to ``process_request`` returning a response object.
    """

    async with aclosing(source):
        try:
            async for item in source:
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            # A downstream close is not a provider failure.  The outer ASGI
            # lifecycle records the client disconnect separately.
            disconnect_event = current_info.get("disconnect_event")
            downstream_disconnected = bool(
                disconnect_event is not None
                and disconnect_event.is_set()
            )
            finalize_latest_routing_attempt(
                current_info,
                response_memory_lease=response_memory_lease,
                outcome=(
                    "downstream_disconnected"
                    if downstream_disconnected
                    else "cancelled_or_consumer_closed"
                ),
                success=None,
            )
            raise
        except BaseException as exc:
            current_info["success"] = False
            local_admission = _record_local_admission_rejection(
                current_info,
                exc,
            )
            transport_failure = _record_transport_failure(
                current_info,
                exc,
                failure_stage="postcommit",
                first_byte_observed=True,
                increment_count=True,
            )
            semantic_failure = isinstance(exc, ResponsesSemanticError)
            if local_admission:
                current_info["stream_outcome"] = "local_backpressure_abort"
                current_info["stream_error_status_code"] = int(
                    getattr(exc, "status_code", 503)
                )
            elif semantic_failure:
                current_info["stream_outcome"] = "upstream_failure_terminal"
                current_info["stream_error_status_code"] = exc.status_code
                current_info["stream_error_code"] = exc.error_code
                current_info["stream_error_type"] = exc.error_type
                current_info["stream_error_event_type"] = exc.event_type
            else:
                current_info["stream_outcome"] = "upstream_stream_abort"
                current_info["stream_error_status_code"] = 502
            current_info["error_type"] = type(exc).__name__
            finalize_latest_routing_attempt(
                current_info,
                response_memory_lease=response_memory_lease,
                outcome=(
                    "semantic_failure_terminal"
                    if semantic_failure
                    else "local_admission_rejected"
                    if local_admission
                    else "stream_failed"
                ),
                success=False,
                semantic_status_code=int(
                    getattr(
                        exc,
                        "status_code",
                        current_info.get("stream_error_status_code") or 502,
                    )
                ),
                terminal_event_type=getattr(exc, "event_type", None),
                error_code=getattr(exc, "error_code", None),
                error_type=(
                    getattr(exc, "error_type", None) or type(exc).__name__
                ),
                error_message=exc,
            )
            if (
                not local_admission
                and (
                    transport_failure is None
                    or transport_failure.provider_penalty_eligible
                )
                and not _is_request_scoped_semantic_error(exc)
            ):
                _schedule_channel_stats_bounded(
                    current_info["request_id"],
                    channel_id,
                    model,
                    current_info["api_key"],
                    success=False,
                    provider_api_key=provider_api_key,
                    fallback_background_tasks=fallback_background_tasks,
                )
            if isinstance(exc, SSEProtocolError):
                # This iterator is already downstream of response.start.  Letting
                # the exception escape through BaseHTTPMiddleware can make its
                # outer body channel reach EOF before Uvicorn sees the exception,
                # allowing the HTTP/1.1 connection to be reused and then reset.
                # Emit one synthetic terminal while this inner iterator still
                # owns the stream, then finish normally so the transport remains
                # isolated from the protocol failure.
                _record_postcommit_sse_protocol_error_isolation(
                    current_info,
                    exc,
                )
                yield _postcommit_sse_protocol_error_event()
                return
            if semantic_failure:
                # Starlette's BaseHTTPMiddleware transports an inner streaming
                # response through an in-memory ASGI channel.  Exceptions that
                # occur after response.start are re-raised only after that
                # channel has reached EOF, which is too late for the outer
                # response wrapper to append a protocol-valid terminal.  Turn
                # a provider-declared Responses failure into the downstream
                # Chat SSE terminal while we still own the body iterator.
                yield (
                    "event: error\n"
                    f"data: {json.dumps(exc.sse_payload, ensure_ascii=False)}\n\n"
                )
                return
            raise
        else:
            current_info["success"] = True
            current_info["provider"] = channel_id
            finalize_latest_routing_attempt(
                current_info,
                response_memory_lease=response_memory_lease,
                outcome="stream_completed",
                success=True,
            )
            _schedule_channel_stats_bounded(
                current_info["request_id"],
                channel_id,
                model,
                current_info["api_key"],
                success=True,
                provider_api_key=provider_api_key,
                fallback_background_tasks=fallback_background_tasks,
            )


def _record_local_admission_rejection(
    current_info: Any,
    exc: BaseException,
) -> bool:
    if not bool(getattr(exc, "local_admission_rejection", False)):
        return False
    if isinstance(current_info, dict):
        reason = str(getattr(exc, "reason", type(exc).__name__))
        current_info["admission_rejected"] = True
        current_info["admission_reason"] = reason
        current_info["error_type"] = reason
        current_info["success"] = False
        current_info["status_origin"] = "ember_local_admission"
    return True


def _record_transport_failure(
    current_info: Any,
    exc: BaseException,
    *,
    failure_stage: Any = None,
    first_byte_observed: bool | None = None,
    increment_count: bool = False,
) -> TransportErrorClassification | None:
    transport_failure = classify_httpx_transport_error(
        exc,
        failure_stage=failure_stage,
        first_byte_observed=first_byte_observed,
    )
    if transport_failure is None:
        return None
    if isinstance(current_info, dict):
        current_info.update(transport_failure.observability_facts())
        current_info["status_origin"] = transport_failure.owner
        if increment_count:
            current_info["transport_error_count"] = max(
                0,
                int(current_info.get("transport_error_count") or 0),
            ) + 1
            if transport_failure.local_overload:
                current_info["local_overload_count"] = max(
                    0,
                    int(current_info.get("local_overload_count") or 0),
                ) + 1
    return transport_failure


def _record_stream_transport_failure(
    current_info: Any,
    attempt: Any,
    exc: BaseException,
) -> TransportErrorClassification | None:
    state = getattr(attempt, "state", None)
    state = state if isinstance(state, dict) else {}
    diagnostics = state.get("responses_stream_diagnostics_tracker")
    diagnostics_facts = getattr(diagnostics, "facts", None)
    if isinstance(diagnostics_facts, dict):
        failure_stage = diagnostics_facts.get("phase") or "postcommit"
        first_byte_observed = bool(
            int(diagnostics_facts.get("upstream_chunk_count") or 0)
        )
    else:
        failure_stage = "postcommit"
        first_byte_observed = True

    transport_failure = _record_transport_failure(
        current_info,
        exc,
        failure_stage=failure_stage,
        first_byte_observed=first_byte_observed,
        increment_count=True,
    )
    if transport_failure is None:
        return None

    facts = transport_failure.observability_facts()
    state.update(facts)
    state["status_origin"] = transport_failure.owner
    routing_entry = state.get("_routing_attempt_entry")
    if isinstance(routing_entry, dict):
        routing_entry.update(facts)
    if not transport_failure.provider_penalty_eligible:
        state["track_channel_stats"] = False
    return transport_failure


def _is_request_scoped_semantic_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, ResponsesSemanticError)
        and exc.status_code in (400, 413)
    )


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
    response_memory_lease: Any = None,
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
    apply_oaix_routing_attempt_id(
        headers,
        provider=provider,
        routing_attempt_id=provider.get("_routing_attempt_id"),
    )
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
            defer_channel_result = False
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
                wrapped_generator = _track_legacy_stream_outcome(
                    wrapped_generator,
                    current_info=current_info,
                    channel_id=channel_id,
                    model=request.model,
                    provider_api_key=provider_api_key_raw,
                    fallback_background_tasks=background_tasks,
                    response_memory_lease=response_memory_lease,
                )
                defer_channel_result = True
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
                setattr(
                    response,
                    "_uni_api_response_attempt_terminal_outcome",
                    "succeeded",
                )
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
                    decoded_element = await run_json_cpu(json.loads, first_element)
                    encoded_element = await run_json_cpu(json.dumps, decoded_element)
                    response = StarletteStreamingResponse(iter([encoded_element]), media_type="application/json", headers=upstream_response_headers)
                    setattr(
                        response,
                        "_uni_api_response_attempt_terminal_outcome",
                        "succeeded",
                    )

            # 更新成功计数和首次响应时间
            if not defer_channel_result:
                _schedule_channel_stats_bounded(
                    current_info["request_id"],
                    channel_id,
                    request.model,
                    current_info["api_key"],
                    success=True,
                    provider_api_key=provider_api_key_raw,
                    fallback_background_tasks=background_tasks,
                )
            current_info["first_response_time"] = first_response_time
            if not defer_channel_result:
                current_info["success"] = True
                current_info["provider"] = channel_id
            setattr(response, "current_info", current_info)
            return response

    except (Exception, HTTPException, asyncio.CancelledError, httpx.ReadError, httpx.RemoteProtocolError, httpx.LocalProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
        disconnect_event = current_info.get("disconnect_event")
        local_admission_rejection = _record_local_admission_rejection(
            current_info,
            e,
        )
        transport_failure = _record_transport_failure(
            current_info,
            e,
            first_byte_observed=bool(
                current_info.get("_first_byte_observed")
            ),
        )
        if not (
            local_admission_rejection
            or (
                transport_failure is not None
                and not transport_failure.provider_penalty_eligible
            )
            or _is_request_scoped_semantic_error(e)
            or isinstance(e, asyncio.CancelledError)
            or (
                isinstance(disconnect_event, asyncio.Event)
                and disconnect_event.is_set()
            )
        ):
            _schedule_channel_stats_bounded(
                current_info["request_id"],
                channel_id,
                request.model,
                current_info["api_key"],
                success=False,
                provider_api_key=provider_api_key_raw,
                fallback_background_tasks=background_tasks,
            )
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
        http_request: Optional[Request] = None,
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
        request_body_bytes = _request_body_size_bytes(http_request, request_data)
        routing_endpoint = endpoint or "/v1/chat/completions"
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=routing_endpoint,
            request_total_tokens=request_total_tokens,
            request_body_bytes=request_body_bytes,
            reasoning_effort=request_reasoning_effort(request_data),
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
            observability_context=current_info,
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
                    request_body_bytes=request_body_bytes,
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
                    request_type=plan.request_type,
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
            attempt_provider = dict(provider)
            attempt_provider["_routing_attempt_id"] = (
                attempt.routing_attempt_id
            )
            process_task = asyncio.create_task(
                process_request(
                    request_data,
                    attempt_provider,
                    background_tasks,
                    endpoint,
                    plan.role,
                    local_timeout_value,
                    keepalive_interval,
                    provider_api_key_raw=attempt.provider_api_key_raw,
                    current_info=current_info,
                    http_request=http_request,
                    response_memory_lease=attempt.state.get(
                        "_response_memory_lease"
                    ),
                )
            )
            disconnect_task: Optional[asyncio.Task] = None
            process_result_transferred = False
            process_cleanup_completed = False

            async def cleanup_abandoned_process_result(result: Any) -> None:
                body_iterator = getattr(result, "body_iterator", None)
                if body_iterator is not None and hasattr(body_iterator, "aclose"):
                    await call_cleanup_safely(
                        body_iterator.aclose,
                        label="Abandoned model response body iterator",
                    )

            try:
                if disconnect_event is not None:
                    disconnect_task = asyncio.create_task(disconnect_event.wait())
                    done, pending = await asyncio.wait(
                        [process_task, disconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if process_task in done:
                        result = process_task.result()
                        # FIRST_COMPLETED may report both tasks in the same
                        # event-loop turn.  The disconnect owns that race: do
                        # not transfer a response whose client is already gone.
                        if disconnect_event.is_set():
                            await cleanup_abandoned_process_result(result)
                            process_cleanup_completed = True
                            return Response(content="", status_code=499)
                        process_result_transferred = True
                        return result
                    if disconnect_task in done and disconnect_event.is_set():
                        await _cancel_awaitable_task_and_cleanup_result(
                            process_task,
                            cleanup_abandoned_process_result,
                        )
                        process_cleanup_completed = True
                        return Response(content="", status_code=499)

                result = await process_task
                process_result_transferred = True
                return result
            except asyncio.CancelledError:
                raise
            except Exception:
                if disconnect_event is not None and disconnect_event.is_set():
                    return Response(content="", status_code=499)
                raise
            finally:
                if not process_result_transferred and not process_cleanup_completed:
                    await _cancel_awaitable_task_and_cleanup_result(
                        process_task,
                        cleanup_abandoned_process_result,
                    )
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

        def should_cool_down(exc, status_code, error_message, attempt):
            _ = error_message, attempt
            return not isinstance(exc, ValueError) and status_code not in (400, 413)

        def build_error_response(status_code, error_message):
            if isinstance(current_info, dict):
                current_info["first_response_time"] = -1
                current_info["success"] = False
                current_info["provider"] = None
            return build_upstream_error_response(
                status_code=status_code,
                error_message=error_message,
                fallback_prefix="Error: Current provider response failed",
            )

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
            build_error_response=build_error_response,
            build_final_response=build_final_response,
            exclude_error_substrings=exclude_error_rate_limit,
            rollback_rate_limit_errors=exclude_error_rate_limit,
            allow_channel_exclusion=True,
            should_cool_down=should_cool_down,
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

_RESPONSE_HEADER_NAME_RE = re.compile(r"[-!#$%&'*+.^_`|~0-9a-zA-Z]+\Z")
_RESPONSE_HEADER_VALUE_RE = re.compile(r"([^\x00\s]+(?:[ \t]+[^\x00\s]+)*)?\Z")


def _response_header_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="replace")
    return str(value)


def _iter_upstream_response_header_pairs(headers: Any):
    raw_headers = getattr(headers, "raw", None)
    if raw_headers:
        for key, value in raw_headers:
            yield _response_header_text(key), _response_header_text(value)
        return
    for key, value in headers.items():
        yield str(key), str(value)


def _is_valid_downstream_response_header(name: str, value: str) -> bool:
    if _RESPONSE_HEADER_NAME_RE.fullmatch(name) is None:
        return False
    if _RESPONSE_HEADER_VALUE_RE.fullmatch(value) is None:
        return False
    try:
        name.encode("latin-1")
        value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


def _log_dropped_upstream_response_header(name: str, reason: str) -> None:
    safe_name = name if _RESPONSE_HEADER_NAME_RE.fullmatch(name) else "<invalid>"
    if reason == "empty_value":
        trace_logger.debug("dropped upstream response header name=%s reason=%s", safe_name, reason)
        return
    trace_logger.warning("dropped upstream response header name=%s reason=%s", safe_name, reason)


def _copy_upstream_response_headers(headers: Any) -> dict[str, str]:
    grouped: dict[str, tuple[str, list[str]]] = {}
    if not headers:
        return {}
    for raw_key, raw_value in _iter_upstream_response_header_pairs(headers):
        name = str(raw_key)
        normalized_key = name.lower()
        if normalized_key in HOP_BY_HOP_RESPONSE_HEADERS:
            continue
        value = str(raw_value).strip(" \t")
        if not value:
            _log_dropped_upstream_response_header(name, "empty_value")
            continue
        if value.strip(", \t") == "":
            _log_dropped_upstream_response_header(name, "empty_comma_joined_value")
            continue
        if not _is_valid_downstream_response_header(name, value):
            _log_dropped_upstream_response_header(name, "invalid_name_or_value")
            continue
        if normalized_key not in grouped:
            grouped[normalized_key] = (name, [value])
        else:
            grouped[normalized_key][1].append(value)

    copied: dict[str, str] = {}
    for name, values in grouped.values():
        value = ", ".join(values)
        if not _is_valid_downstream_response_header(name, value):
            _log_dropped_upstream_response_header(name, "invalid_joined_value")
            continue
        copied[name] = value
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


def _upstream_logical_authority(value: Any) -> str:
    try:
        parsed = urlparse(str(value or ""))
        host = parsed.hostname or ""
        port = parsed.port
    except (TypeError, ValueError):
        return ""
    if not host:
        return ""
    rendered_host = f"[{host}]" if ":" in host else host
    return f"{rendered_host}:{port}" if port is not None else rendered_host


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


def _resolve_alpha_search_timeout(
    *,
    provider_name: str,
    original_model: str,
    request_model: str,
    role: str,
    engine: str,
) -> Optional[httpx.Timeout]:
    timeout_value = get_preference(
        app.state.provider_timeouts,
        provider_name,
        (original_model, request_model),
        DEFAULT_TIMEOUT,
    )
    timeout_resolution = apply_timeout_policy(
        base_timeout=int(timeout_value),
        timeout_policy=getattr(app.state, "timeout_policy", {}),
        provider_name=provider_name,
        endpoint=ALPHA_SEARCH_ENDPOINT,
        method="POST",
        stream=False,
        engine=engine,
        original_model=original_model,
        request_model=request_model,
        role=role,
    )
    return _httpx_timeout_from_policy(timeout_resolution, stream=False)


class DownstreamDisconnectedDuringWait(Exception):
    """The downstream peer left while an upstream operation was pending."""


def _upstream_read_timeout(message: str, *, timeout_seconds: float) -> httpx.ReadTimeout:
    seconds = float(timeout_seconds)
    request = httpx.Request(
        "POST",
        "https://uni-api.local/upstream-timeout",
        extensions={"timeout": {"read": seconds}},
    )
    return httpx.ReadTimeout(message, request=request)


async def _await_first_byte_deadline(
    awaitable: Awaitable[Any],
    *,
    timeout_seconds: Any = None,
    deadline: Optional[float] = None,
    total_timeout_seconds: Any = None,
    total_deadline: Optional[float] = None,
    satisfied: Optional[Callable[[], bool]] = None,
    cancel_result_cleanup: Optional[Callable[[Any], Awaitable[None]]] = None,
    disconnect_event: Optional[asyncio.Event] = None,
) -> Any:
    timeout = _optional_positive_timeout(timeout_seconds)
    total_timeout = _optional_positive_timeout(total_timeout_seconds)
    if deadline is None:
        if timeout is None:
            if total_deadline is None and disconnect_event is None:
                return await awaitable
        else:
            deadline = asyncio.get_running_loop().time() + timeout
    if deadline is None and total_deadline is None and disconnect_event is None:
        return await awaitable
    task = asyncio.create_task(awaitable)
    disconnect_task: Optional[asyncio.Task[bool]] = None
    if disconnect_event is not None:
        disconnect_task = asyncio.create_task(disconnect_event.wait())
    loop = asyncio.get_running_loop()
    first_byte_timeout_for_message = timeout if timeout is not None else (
        max(0.0, deadline - loop.time()) if deadline is not None else None
    )
    total_timeout_for_message = total_timeout if total_timeout is not None else (
        max(0.0, total_deadline - loop.time()) if total_deadline is not None else None
    )
    task_result_owned = True
    try:
        while True:
            first_byte_satisfied = satisfied is not None and satisfied()
            active_deadlines: list[tuple[str, float, Optional[float]]] = []
            if not first_byte_satisfied and deadline is not None:
                active_deadlines.append(("first byte", deadline, first_byte_timeout_for_message))
            if total_deadline is not None:
                active_deadlines.append(("total response", total_deadline, total_timeout_for_message))
            if active_deadlines:
                timeout_label, active_deadline, timeout_for_message = min(
                    active_deadlines,
                    key=lambda item: item[1],
                )
                remaining = active_deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError(timeout_label)
                wait_timeout: Optional[float] = min(0.05, remaining)
            else:
                wait_timeout = None
            wait_tasks: set[asyncio.Task[Any]] = {task}
            if disconnect_task is not None:
                wait_tasks.add(disconnect_task)
            done, _ = await asyncio.wait(
                wait_tasks,
                timeout=wait_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # A completed upstream operation owns the loop turn even when the
            # peer disconnect arrives simultaneously.  Its caller performs a
            # final disconnect check before committing the response.
            if task in done:
                result = task.result()
                task_result_owned = False
                return result
            if (
                disconnect_task is not None
                and disconnect_task in done
                and disconnect_event is not None
                and disconnect_event.is_set()
            ):
                raise DownstreamDisconnectedDuringWait()
    except asyncio.TimeoutError as exc:
        if task_result_owned:
            await _cancel_awaitable_task_and_cleanup_result(
                task,
                cancel_result_cleanup,
            )
        timeout_label = exc.args[0] if exc.args else "first byte"
        timeout_for_message = (
            total_timeout_for_message if timeout_label == "total response" else first_byte_timeout_for_message
        )
        timeout_for_message = timeout_for_message if timeout_for_message is not None else 0
        raise _upstream_read_timeout(
            f"Request timed out waiting for {timeout_label} after {timeout_for_message:g} seconds",
            timeout_seconds=timeout_for_message,
        ) from exc
    except BaseException:
        if task_result_owned:
            await _cancel_awaitable_task_and_cleanup_result(
                task,
                cancel_result_cleanup,
            )
        raise
    finally:
        if disconnect_task is not None and not disconnect_task.done():
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task


async def _cancel_awaitable_task_and_cleanup_result(
    task: asyncio.Task[Any],
    cleanup_result: Optional[Callable[[Any], Awaitable[None]]],
) -> None:
    if not task.done():
        task.cancel()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    try:
        result = task.result()
    except (asyncio.CancelledError, Exception):
        return
    if cleanup_result is None:
        return
    cleanup_task = asyncio.create_task(cleanup_result(result))
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            continue
    cleanup_task.result()


async def _await_buffered_upstream_or_disconnect(
    awaitable: Awaitable[Any],
    disconnect_event: Optional[asyncio.Event],
) -> Any:
    async def close_late_response(response: Any) -> None:
        close = getattr(response, "aclose", None)
        if callable(close):
            await close()

    result = await _await_first_byte_deadline(
        awaitable,
        disconnect_event=disconnect_event,
        cancel_result_cleanup=close_late_response,
    )
    if disconnect_event is not None and disconnect_event.is_set():
        await close_late_response(result)
        raise DownstreamDisconnectedDuringWait()
    return result


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
        raise _upstream_read_timeout(
            f"Request timed out waiting for total response after {float(total_timeout_seconds):g} seconds",
            timeout_seconds=float(total_timeout_seconds),
        )
    try:
        return await asyncio.wait_for(upstream_iter.__anext__(), timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise _upstream_read_timeout(
            f"Request timed out waiting for total response after {float(total_timeout_seconds):g} seconds",
            timeout_seconds=float(total_timeout_seconds),
        ) from exc

async def _prime_passthrough_upstream_stream(
    upstream_iter,
    *,
    disconnect_event: Optional[asyncio.Event] = None,
) -> list[bytes]:
    while True:
        if disconnect_event is not None and disconnect_event.is_set():
            return []

        try:
            chunk = await _await_first_byte_deadline(
                upstream_iter.__anext__(),
                disconnect_event=disconnect_event,
            )
        except StopAsyncIteration:
            raise HTTPException(status_code=502, detail="Upstream closed stream without data")

        if not chunk:
            continue
        chunk_bytes = bytes(chunk)
        if len(chunk_bytes) > DEFAULT_MAX_EVENT_BYTES:
            raise HTTPException(
                status_code=502,
                detail="Upstream transport chunk exceeded local stream limit",
            )
        return [chunk_bytes]

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
RESPONSES_DELTA_FAST_PATH_ENABLED = _env_bool(
    "RESPONSES_DELTA_FAST_PATH_ENABLED",
    True,
)

RESPONSES_STREAM_QUEUE_MAX_ITEMS = max(
    1,
    _env_int("RESPONSES_STREAM_QUEUE_MAX_ITEMS", 64),
)
_responses_stream_transient_guard_bytes = max(
    1,
    _startup_memory_snapshot.guard_bytes // 2,
)
_responses_stream_default_queue_bytes = max(
    64 * 1024,
    min(
        2 * 1024 * 1024,
        _responses_stream_transient_guard_bytes,
    ),
)
RESPONSES_STREAM_QUEUE_MAX_BYTES = max(
    1,
    _env_int(
        "RESPONSES_STREAM_QUEUE_MAX_BYTES",
        _responses_stream_default_queue_bytes,
    ),
)
RESPONSES_STREAM_QUEUE_PUT_TIMEOUT_SECONDS = max(
    0.001,
    _env_float("RESPONSES_STREAM_QUEUE_PUT_TIMEOUT_SECONDS", 30.0),
)
RESPONSES_STREAM_GLOBAL_BUDGET_BYTES = max(
    1,
    _env_int(
        "RESPONSES_STREAM_GLOBAL_BUDGET_BYTES",
        process_memory_governor.maximum_capacity_bytes(),
    ),
)
# Responses frames have no fixed byte ceiling. JSON validation is instead
# bounded by the current process/cgroup capacity and the request admission
# lease that owns the materialization workspace.
RESPONSES_SSE_JSON_MAX_ESTIMATED_BYTES = (
    process_memory_governor.maximum_capacity_bytes()
)
RESPONSES_STREAM_GLOBAL_BUDGET_WAIT_TIMEOUT_SECONDS = max(
    0.001,
    _env_float(
        "RESPONSES_STREAM_GLOBAL_BUDGET_WAIT_TIMEOUT_SECONDS",
        RESPONSES_STREAM_QUEUE_PUT_TIMEOUT_SECONDS,
    ),
)
RESPONSES_STREAM_PRECOMMIT_MAX_ITEMS = max(
    1,
    _env_int("RESPONSES_STREAM_PRECOMMIT_MAX_ITEMS", 128),
)
# Keep a bounded aggregate structural precommit buffer. A frame that commits
# the response bypasses this fixed threshold and remains memory-accounted.
RESPONSES_CANONICAL_EVENT_MAX_OVERHEAD_BYTES = (
    RESPONSES_CANONICAL_EVENT_HEADER_MAX_BYTES + len(b"\n\n")
)
RESPONSES_STREAM_PRECOMMIT_MAX_BYTES = max(
    1,
    _env_int(
        "RESPONSES_STREAM_PRECOMMIT_MAX_BYTES",
        DEFAULT_MAX_EVENT_BYTES
        + RESPONSES_STREAM_PRECOMMIT_MAX_ITEMS
        * RESPONSES_CANONICAL_EVENT_MAX_OVERHEAD_BYTES,
    ),
)

responses_stream_byte_budget = RetainedByteBudget(
    capacity_bytes=RESPONSES_STREAM_GLOBAL_BUDGET_BYTES,
    wait_timeout_seconds=RESPONSES_STREAM_GLOBAL_BUDGET_WAIT_TIMEOUT_SECONDS,
    memory_governor=process_memory_governor,
)
runtime_gauges.attach_stream_byte_budget(responses_stream_byte_budget.snapshot)
RESPONSES_STREAM_STATS_QUEUE_MAX_ITEMS = max(
    1,
    _env_int("RESPONSES_STREAM_STATS_QUEUE_MAX_ITEMS", 1024),
)
RESPONSES_STREAM_STATS_WORKERS = max(
    1,
    _env_int("RESPONSES_STREAM_STATS_WORKERS", 4),
)
responses_stream_stats_queue: Optional[
    asyncio.Queue[tuple[tuple[Any, ...], dict[str, Any]]]
] = None
responses_stream_stats_workers: set[asyncio.Task[None]] = set()
responses_stream_stats_submitted = 0
responses_stream_stats_completed = 0
responses_stream_stats_failed = 0
responses_stream_stats_dropped = 0


def _responses_stream_stats_snapshot() -> dict[str, Any]:
    queue = responses_stream_stats_queue
    return {
        "items": queue.qsize() if queue is not None else 0,
        "capacity": RESPONSES_STREAM_STATS_QUEUE_MAX_ITEMS,
        "workers": len(responses_stream_stats_workers),
        "submitted": responses_stream_stats_submitted,
        "completed": responses_stream_stats_completed,
        "failed": responses_stream_stats_failed,
        "dropped": responses_stream_stats_dropped,
    }


runtime_gauges.attach_stream_stats(_responses_stream_stats_snapshot)


async def _responses_stream_stats_worker(worker_index: int) -> None:
    global responses_stream_stats_completed, responses_stream_stats_failed
    queue = responses_stream_stats_queue
    if queue is None:
        return
    while True:
        args, kwargs = await queue.get()
        try:
            persisted = await update_channel_stats(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            responses_stream_stats_failed += 1
            logger.exception(
                "Responses stream stats worker %d failed to persist update",
                worker_index,
            )
        else:
            if persisted is False:
                responses_stream_stats_failed += 1
                continue
            responses_stream_stats_completed += 1
        finally:
            queue.task_done()


async def _start_responses_stream_stats_workers() -> None:
    global responses_stream_stats_queue
    if responses_stream_stats_workers:
        return
    responses_stream_stats_queue = asyncio.Queue(
        maxsize=RESPONSES_STREAM_STATS_QUEUE_MAX_ITEMS
    )
    for worker_index in range(RESPONSES_STREAM_STATS_WORKERS):
        task = asyncio.create_task(
            _responses_stream_stats_worker(worker_index),
            name=f"uni-api-responses-stream-stats-{worker_index}",
        )
        responses_stream_stats_workers.add(task)


def _enqueue_responses_stream_stats(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    global responses_stream_stats_submitted, responses_stream_stats_dropped
    queue = responses_stream_stats_queue
    if queue is None:
        return False
    try:
        queue.put_nowait((args, kwargs))
    except asyncio.QueueFull:
        responses_stream_stats_dropped += 1
        if responses_stream_stats_dropped & (responses_stream_stats_dropped - 1) == 0:
            logger.warning(
                "Channel stats queue full; dropped_updates=%d",
                responses_stream_stats_dropped,
            )
        return False
    responses_stream_stats_submitted += 1
    return True


def _schedule_channel_stats_bounded(
    *args: Any,
    fallback_background_tasks: Optional[BackgroundTasks] = None,
    **kwargs: Any,
) -> None:
    if responses_stream_stats_queue is not None:
        _enqueue_responses_stream_stats(tuple(args), dict(kwargs))
        return
    # Unit-level handler tests may execute without the application lifespan.
    # Production always has the bounded worker queue started before serving.
    if fallback_background_tasks is not None:
        fallback_background_tasks.add_task(update_channel_stats, *args, **kwargs)


async def _stop_responses_stream_stats_workers(*, timeout: float) -> None:
    global responses_stream_stats_queue
    queue = responses_stream_stats_queue
    if queue is None:
        return
    try:
        await asyncio.wait_for(queue.join(), timeout=max(0.001, timeout))
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out draining Responses stream stats queue: pending=%d",
            queue.qsize(),
        )
    workers = list(responses_stream_stats_workers)
    for task in workers:
        task.cancel()
    if workers:
        await asyncio.gather(*workers, return_exceptions=True)
    responses_stream_stats_workers.clear()
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        else:
            queue.task_done()
    responses_stream_stats_queue = None

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


@dataclass(frozen=True, slots=True)
class _StreamReceiveTiming:
    received_at: datetime
    unix_nano: int
    monotonic_nano: int


def _capture_stream_receive_timing() -> _StreamReceiveTiming:
    unix_nano = time_ns()
    return _StreamReceiveTiming(
        received_at=datetime.fromtimestamp(
            unix_nano / 1_000_000_000,
            tz=timezone.utc,
        ),
        unix_nano=unix_nano,
        monotonic_nano=perf_counter_ns(),
    )


def _canonical_responses_sse_event_bytes(
    raw_event: str,
    *,
    event_type: str,
    has_event_field: bool,
    wire_bytes: bytes,
) -> tuple[bytes, bool]:
    """Emit one data-bearing Responses event in a stable SSE shape.

    A data-only event is valid SSE and its Responses type lives in the JSON
    payload.  Add the redundant event field for downstream SDK compatibility.
    Leading blank lines are transport no-ops left by repeated separators and
    are removed from the canonical frame.
    """

    if raw_event.startswith("\n"):
        normalized_wire = wire_bytes[1:]
        normalized = True
    else:
        # Preserve object identity on the common path so diagnostics, the
        # terminal hash, and downstream delivery share the one UTF-8 encoding.
        normalized_wire = wire_bytes
        normalized = False
    if not has_event_field and event_type and event_type != "[DONE]":
        encoded_event_type = event_type.encode("utf-8")
        if (
            len(encoded_event_type) > 256
            or b"\r" in encoded_event_type
            or b"\n" in encoded_event_type
        ):
            raise SSEProtocolError("Responses upstream event type is invalid")
        normalized_wire = (
            b"event: " + encoded_event_type + b"\n" + normalized_wire
        )
        normalized = True
    return normalized_wire, normalized


def _observed_responses_stream_chunk(
    data: bytes,
    reservation: Any | None,
    *,
    event_type: str,
    semantic_outcome: str,
    terminal_timeline_event: bool = False,
    usage_snapshot: StreamUsageSnapshot | None = None,
) -> ObservedStreamFrame | ReservedStreamChunk:
    if reservation is not None:
        return ReservedStreamChunk(
            data,
            reservation,
            event_type=event_type,
            semantic_outcome=semantic_outcome,
            terminal_timeline_event=terminal_timeline_event,
            sse_metadata_complete=True,
            usage_snapshot=usage_snapshot,
        )
    return ObservedStreamFrame(
        data=data,
        event_type=event_type,
        semantic_outcome=semantic_outcome,
        terminal_timeline_event=terminal_timeline_event,
        sse_metadata_complete=True,
        usage_snapshot=usage_snapshot,
    )


async def _consume_expected_oaix_terminal_flush_marker(
    source: Any,
    diagnostics: ResponsesStreamDiagnostics,
    *,
    timeout_seconds: float = 0.25,
    max_frames: int = 4,
) -> None:
    """Consume OAIX's post-terminal diagnostic comment without forwarding it.

    The terminal has already been handed to the downstream queue. OAIX writes
    this marker immediately after its successful terminal flush. A strict,
    short bound keeps a missing observability marker from extending a business
    request or holding an upstream connection indefinitely.
    """

    if not diagnostics.expects_oaix_terminal_flush_marker:
        return
    if diagnostics.terminal_hop_observed:
        return
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.01, float(timeout_seconds))
    for _ in range(max(1, int(max_frames))):
        remaining = deadline - loop.time()
        if remaining <= 0:
            diagnostics.mark_terminal_flush_marker_missing("timeout")
            return
        raw_event = None
        reservation = None
        receive_timing = None
        try:
            (
                raw_event,
                reservation,
                _from_precommit,
                already_observed,
                receive_timing,
                _upstream_event_bytes,
            ) = await asyncio.wait_for(
                source.__anext__(),
                timeout=remaining,
            )
        except StopAsyncIteration:
            diagnostics.mark_terminal_flush_marker_missing("upstream_eof")
            return
        except asyncio.TimeoutError:
            diagnostics.mark_terminal_flush_marker_missing("timeout")
            return
        try:
            if not already_observed:
                diagnostics.observe_complete_event(
                    raw_event,
                    has_data_field=sse_event_has_data_field(raw_event),
                    received_at=(
                        receive_timing.received_at
                        if receive_timing is not None
                        else None
                    ),
                    received_unix_nano=(
                        receive_timing.unix_nano
                        if receive_timing is not None
                        else None
                    ),
                    received_monotonic_nano=(
                        receive_timing.monotonic_nano
                        if receive_timing is not None
                        else None
                    ),
                )
            if is_oaix_terminal_flush_marker(raw_event):
                if not diagnostics.terminal_hop_observed:
                    diagnostics.mark_terminal_flush_marker_missing(
                        "invalid_marker"
                    )
                return
            if not is_sse_comment_frame(raw_event):
                diagnostics.mark_terminal_flush_marker_missing(
                    "unexpected_post_terminal_event"
                )
                return
        finally:
            raw_event = None
            if reservation is not None:
                await reservation.release()
                reservation = None
    diagnostics.mark_terminal_flush_marker_missing("frame_limit")

def _build_responses_stream_keepalive_event() -> bytes:
    return _encode_responses_sse_event(
        "keepalive",
        {"type": "keepalive", "sequence_number": 0},
    )


def _is_oaix_precommit_keepalive(chunk: bytes) -> bool:
    # The canonical OAIX keepalive is under 100 bytes.  Never synchronously
    # reparse an attacker-sized upstream event merely to recognize it.
    if len(chunk) > 1024:
        return False
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
                "message": bounded_stream_error_text(error_message),
                "status_code": int(status_code),
            },
        },
    )


def _observed_responses_stream_error_event(
    status_code: int,
    error_message: Any,
) -> ObservedStreamChunk:
    return ObservedStreamChunk(
        _build_responses_stream_error_event(status_code, error_message),
        event_type="error",
        semantic_outcome="error",
        sse_metadata_complete=True,
    )

def _stream_error_event_from_response(response: Any) -> ObservedStreamChunk:
    status_code = int(getattr(response, "status_code", 500) or 500)
    body = getattr(response, "body", b"")
    message = bounded_stream_error_text(
        body or f"Upstream request failed with status {status_code}"
    )
    return _observed_responses_stream_error_event(status_code, message)

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

def _responses_precommit_semantic_guard(engine: str) -> bool:
    return engine == "codex"


def _responses_stream_event_commits(
    event_type: str,
    payload: Any,
    commit_policy: str,
    *,
    precommit_semantic_guard: bool = True,
) -> bool:
    if event_type in RESPONSES_STREAM_PREFLIGHT_EVENTS:
        return not precommit_semantic_guard and event_type == "response.created"

    if event_type in {"response.completed", "response.incomplete"}:
        return True

    if commit_policy == "completed_usage":
        return False
    return _responses_stream_event_has_real_output(event_type, payload)


def _validate_responses_terminal_payload(event_type: str, payload: Any) -> None:
    """Reject terminal labels whose data is not a Responses object."""

    if event_type not in {
        "error",
        "response.completed",
        "response.failed",
        "response.incomplete",
    }:
        return
    if not isinstance(payload, dict):
        raise SSEProtocolError(
            f"Responses upstream {event_type} payload must be a JSON object"
        )
    if event_type.startswith("response.") and not isinstance(
        payload.get("response"), dict
    ):
        raise SSEProtocolError(
            f"Responses upstream {event_type} payload is missing response"
        )
    if event_type == "error" and payload.get("error") is None:
        raise SSEProtocolError(
            "Responses upstream error payload is missing error"
        )

def _responses_failure_http_exception(
    payload: Any,
    *,
    event_type: Optional[str] = None,
    wire_status_code: Optional[int] = None,
    validated_provider_sse: bool = False,
) -> Optional[ResponsesSemanticError]:
    normalized = responses_failure_error(
        payload,
        event_type=event_type,
        wire_status_code=wire_status_code,
        preserve_error_body=True,
        validated_provider_sse=validated_provider_sse,
    )
    if normalized is not None:
        return normalized

    # Preserve the legacy fallback for unusual non-terminal JSON payloads
    # that contain an explicit error object but no event/status discriminator.
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

    fallback_payload = {
        "type": "error",
        "error": error_obj,
    }
    return responses_failure_error(
        fallback_payload,
        event_type="error",
        wire_status_code=wire_status_code,
        preserve_error_body=True,
    )

async def _prime_responses_upstream_stream(
    upstream_iter,
    *,
    upstream_status_code: Optional[int] = None,
    disconnect_event: Optional[asyncio.Event] = None,
    commit_policy: str = "real_output",
    precommit_semantic_guard: bool = True,
    precommit_keepalive_callback: Optional[Callable[[Optional[bytes]], Awaitable[bool]]] = None,
    retained_byte_budget: RetainedByteBudget | None = None,
    diagnostics: ResponsesStreamDiagnostics | None = None,
) -> tuple[ReservedChunkBuffer, bool]:
    """
    With the semantic guard enabled, buffer structural Responses events until
    substantive output or a completed response with usage. Transparent
    providers commit on response.created and never synthesize a keepalive.
    """
    buffered_chunks = ReservedChunkBuffer(
        max_items=RESPONSES_STREAM_PRECOMMIT_MAX_ITEMS,
        max_bytes=RESPONSES_STREAM_PRECOMMIT_MAX_BYTES,
        retained_byte_budget=retained_byte_budget,
    )
    sse_parser = IncrementalSSEParser(
        max_pending_bytes=UNLIMITED_SSE_BYTES,
        max_event_bytes=UNLIMITED_SSE_BYTES,
        max_feed_bytes=UNLIMITED_SSE_BYTES,
    )
    commit_policy = (commit_policy or "real_output").strip().lower()
    if commit_policy not in {"real_output", "completed_usage"}:
        commit_policy = "real_output"

    async def append_buffered(
        chunk: bytes,
        *,
        allow_over_limit: bool = False,
    ) -> None:
        try:
            await buffered_chunks.append(
                chunk,
                allow_over_limit=allow_over_limit,
            )
        except StreamQueueItemTooLarge as exc:
            raise HTTPException(
                status_code=502,
                detail="Responses upstream precommit buffer limit exceeded",
            ) from exc

    chunk = None
    raw_event = None
    raw_events = []
    raw_event_wires: list[bytes | None] = []
    raw_events_receive_timing: _StreamReceiveTiming | None = None
    phase_diagnostics = (
        diagnostics
        if diagnostics is not None and diagnostics.phase_sampled
        else None
    )

    def observe_pending_diagnostics() -> None:
        if diagnostics is None:
            return
        pending = sse_parser.failure_pending_diagnostics
        diagnostics.observe_partial_diagnostics(
            pending if pending is not None else sse_parser.pending_diagnostics()
        )

    try:
        while True:
            if disconnect_event is not None and disconnect_event.is_set():
                observe_pending_diagnostics()
                return buffered_chunks, False

            reached_eof = False
            receive_phase = (
                phase_diagnostics.begin_phase("socket_receive")
                if phase_diagnostics is not None
                else None
            )
            try:
                chunk = await _await_first_byte_deadline(
                    upstream_iter.__anext__(),
                    disconnect_event=disconnect_event,
                )
                raw_events_receive_timing = _capture_stream_receive_timing()
            except StopAsyncIteration:
                if phase_diagnostics is not None:
                    phase_diagnostics.finish_phase(
                        "socket_receive",
                        receive_phase,
                    )
                reached_eof = True
                raw_events_receive_timing = _capture_stream_receive_timing()
                observe_pending_diagnostics()
                frame_phase = (
                    phase_diagnostics.begin_phase("sse_frame")
                    if phase_diagnostics is not None
                    else None
                )
                try:
                    raw_events = sse_parser.finish()
                except SSEProtocolError as exc:
                    if diagnostics is not None:
                        if sse_parser.failure_pending_diagnostics is not None:
                            diagnostics.observe_partial_diagnostics(
                                sse_parser.failure_pending_diagnostics
                            )
                        diagnostics.observe_exception(
                            exc,
                            origin="precommit_sse_finish",
                        )
                    raise HTTPException(
                        status_code=502,
                        detail=f"Invalid upstream SSE stream: {exc}",
                    ) from exc
                finally:
                    if phase_diagnostics is not None:
                        phase_diagnostics.finish_phase(
                            "sse_frame",
                            frame_phase,
                            events=(
                                len(raw_events)
                                if isinstance(raw_events, list)
                                else 0
                            ),
                        )
            except BaseException:
                if phase_diagnostics is not None:
                    phase_diagnostics.finish_phase(
                        "socket_receive",
                        receive_phase,
                    )
                raise
            else:
                chunk_size = len(chunk)
                if phase_diagnostics is not None:
                    phase_diagnostics.finish_phase(
                        "socket_receive",
                        receive_phase,
                        bytes_count=chunk_size,
                    )
                frame_phase = (
                    phase_diagnostics.begin_phase("sse_frame")
                    if phase_diagnostics is not None
                    else None
                )
                try:
                    raw_events = sse_parser.feed(chunk)
                except SSEProtocolError as exc:
                    if diagnostics is not None:
                        if sse_parser.failure_pending_diagnostics is not None:
                            diagnostics.observe_partial_diagnostics(
                                sse_parser.failure_pending_diagnostics
                            )
                        diagnostics.observe_exception(
                            exc,
                            origin="precommit_sse_feed",
                        )
                    raise HTTPException(
                        status_code=502,
                        detail=f"Invalid upstream SSE stream: {exc}",
                    ) from exc
                finally:
                    if phase_diagnostics is not None:
                        phase_diagnostics.finish_phase(
                            "sse_frame",
                            frame_phase,
                            bytes_count=chunk_size,
                            events=(
                                len(raw_events)
                                if isinstance(raw_events, list)
                                else 0
                            ),
                        )
                    chunk = None

            raw_event_wires = [
                _raw_responses_sse_event_bytes(observed_event)
                if observed_event.strip()
                else None
                for observed_event in raw_events
            ]
            if diagnostics is not None:
                for observed_event, observed_wire in zip(
                    raw_events,
                    raw_event_wires,
                ):
                    if observed_event.strip():
                        assert observed_wire is not None
                        assert raw_events_receive_timing is not None
                        observer_phase = (
                            phase_diagnostics.begin_phase("observer_hash")
                            if phase_diagnostics is not None
                            else None
                        )
                        try:
                            diagnostics.observe_complete_event(
                                observed_event,
                                has_data_field=sse_event_has_data_field(
                                    observed_event
                                ),
                                wire_bytes=observed_wire,
                                received_at=(
                                    raw_events_receive_timing.received_at
                                ),
                                received_unix_nano=(
                                    raw_events_receive_timing.unix_nano
                                ),
                                received_monotonic_nano=(
                                    raw_events_receive_timing.monotonic_nano
                                ),
                            )
                        finally:
                            if phase_diagnostics is not None:
                                phase_diagnostics.finish_phase(
                                    "observer_hash",
                                    observer_phase,
                                    bytes_count=len(observed_wire),
                                    events=1,
                                )

            for event_index, raw_event in enumerate(raw_events):
                if not raw_event.strip():
                    continue

                parse_phase = (
                    phase_diagnostics.begin_phase("json_parse")
                    if phase_diagnostics is not None
                    else None
                )
                try:
                    event_owner = await parse_owned_sse_event(
                        raw_event,
                        max_event_bytes=UNLIMITED_SSE_BYTES,
                        max_json_estimated_bytes=(
                            RESPONSES_SSE_JSON_MAX_ESTIMATED_BYTES
                        ),
                    )
                finally:
                    if phase_diagnostics is not None:
                        phase_diagnostics.finish_phase(
                            "json_parse",
                            parse_phase,
                            bytes_count=len(raw_event_wires[event_index] or b""),
                            events=1,
                        )
                event_type = event_owner.event_name
                event_payload = event_owner.payload
                if diagnostics is not None:
                    assert raw_events_receive_timing is not None
                    diagnostics.mark_terminal_parse_completed(
                        event_type,
                        received_at=raw_events_receive_timing.received_at,
                        received_unix_nano=raw_events_receive_timing.unix_nano,
                        received_monotonic_nano=(
                            raw_events_receive_timing.monotonic_nano
                        ),
                    )
                semantic_failure = None
                event_bytes = raw_event_wires[event_index]
                assert event_bytes is not None
                handled = None
                try:
                    if event_owner.is_comment:
                        await append_buffered(event_bytes)
                        continue
                    if not event_owner.has_data_field:
                        if diagnostics is not None:
                            diagnostics.observe_normalization(
                                "ignored_no_data_event_block",
                                event_owner.declared_event_name,
                            )
                        continue

                    if event_type == "[DONE]":
                        raise HTTPException(
                            status_code=502,
                            detail="Responses upstream ended before substantive output",
                        )

                    try:
                        validate_sse_event_type_consistency(
                            event_owner.declared_event_name,
                            event_payload,
                            protocol_name="Responses",
                            has_event_field=event_owner.has_event_field,
                            require_event_name=True,
                        )
                        _validate_responses_terminal_payload(
                            event_type,
                            event_payload,
                        )
                    except SSEProtocolError as exc:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Invalid upstream SSE stream: {exc}",
                        ) from exc
                    semantic_failure = _responses_failure_http_exception(
                        event_payload,
                        event_type=event_type,
                        wire_status_code=upstream_status_code,
                        validated_provider_sse=True,
                    )
                    if semantic_failure is not None:
                        semantic_outcome = "failed"
                    elif event_type == "response.completed":
                        semantic_outcome = "completed"
                    elif event_type == "response.incomplete":
                        semantic_outcome = "incomplete"
                    else:
                        semantic_outcome = "nonterminal"
                    if diagnostics is not None:
                        observer_phase = (
                            phase_diagnostics.begin_phase("observer_hash")
                            if phase_diagnostics is not None
                            else None
                        )
                        try:
                            diagnostics.observe_parsed_event(
                                raw_event,
                                event_type,
                                event_payload,
                                semantic_outcome=semantic_outcome,
                                wire_bytes=event_bytes,
                                received_at=(
                                    raw_events_receive_timing.received_at
                                ),
                            )
                        finally:
                            if phase_diagnostics is not None:
                                phase_diagnostics.finish_phase(
                                    "observer_hash",
                                    observer_phase,
                                )
                    if semantic_failure is not None:
                        if (
                            event_type == "error"
                            and semantic_failure.responses_sse_event_type
                            == "response.failed"
                            and diagnostics is not None
                        ):
                            diagnostics.observe_normalization(
                                "provider_error_to_response_failed",
                                event_type,
                            )
                        raise semantic_failure

                    event_bytes, _event_was_normalized = (
                        _canonical_responses_sse_event_bytes(
                            raw_event,
                            event_type=event_type,
                            has_event_field=event_owner.has_event_field,
                            wire_bytes=event_bytes,
                        )
                    )
                    if (
                        not event_owner.has_event_field
                        and event_type
                        and event_type != "[DONE]"
                        and diagnostics is not None
                    ):
                        diagnostics.observe_normalization(
                            "canonicalized_data_only_event",
                            event_type,
                        )
                    commits_response = _responses_stream_event_commits(
                        event_type,
                        event_payload,
                        commit_policy,
                        precommit_semantic_guard=precommit_semantic_guard,
                    )
                    if (
                        precommit_semantic_guard
                        and event_type == "keepalive"
                        and precommit_keepalive_callback is not None
                    ):
                        handled = await precommit_keepalive_callback(event_bytes)
                        if not handled:
                            await append_buffered(event_bytes)
                    else:
                        if (
                            precommit_semantic_guard
                            and event_type == "response.created"
                            and precommit_keepalive_callback is not None
                        ):
                            await precommit_keepalive_callback(None)
                        await append_buffered(
                            event_bytes,
                            allow_over_limit=commits_response,
                        )

                    if commits_response:
                        for remaining_index in range(
                            event_index + 1,
                            len(raw_events),
                        ):
                            remaining_raw_event = raw_events[remaining_index]
                            if remaining_raw_event.strip():
                                remaining_wire = raw_event_wires[remaining_index]
                                assert remaining_wire is not None
                                await append_buffered(
                                    remaining_wire,
                                    allow_over_limit=True,
                                )
                        if sse_parser.pending_data:
                            await append_buffered(
                                sse_parser.pending_data,
                                allow_over_limit=True,
                            )
                        return buffered_chunks, True
                finally:
                    # The owned parser reservation may only be returned after
                    # every local alias into its materialized graph is gone.
                    handled = None
                    event_bytes = None
                    semantic_failure = None
                    event_payload = None
                    event_type = None
                    await event_owner.aclose()

            # IncrementalSSEParser frames carry process-wide retained-byte
            # ownership.  A completed nonterminal batch must not remain live
            # while the next upstream byte is awaited.
            raw_event = None
            raw_events.clear()
            raw_events = None
            raw_event_wires.clear()
            raw_event_wires = []
            raw_events_receive_timing = None

            if reached_eof:
                if not buffered_chunks:
                    raise HTTPException(
                        status_code=502,
                        detail="Upstream closed stream without data",
                    )
                raise HTTPException(
                    status_code=502,
                    detail="Responses upstream closed before substantive output",
                )
    except BaseException:
        observe_pending_diagnostics()
        sse_parser.discard()
        chunk = None
        raw_event = None
        if raw_events is not None:
            raw_events.clear()
        raw_event_wires.clear()
        await buffered_chunks.clear()
        raise
    finally:
        chunk = None
        raw_event = None
        if raw_events is not None:
            raw_events.clear()
        raw_events = None
        raw_event_wires.clear()
        raw_events_receive_timing = None
        sse_parser.discard()

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
        if (
            endpoint == "/v1/responses"
            and bool(request_data.stream)
            and RUST_RESPONSES_DATA_PLANE_ENABLED
            and _has_valid_rust_control_token(http_request)
            and not http_request.headers.get("idempotency-key")
        ):
            execution.current_info["rust_responses_data_plane"] = True
            execution.current_info["rust_responses_control_version"] = 1
            session = await rust_responses_control_plane.create(execution)

            async def control_body() -> AsyncIterator[bytes]:
                try:
                    async for chunk in session.control_body():
                        yield chunk
                finally:
                    await rust_responses_control_plane.release(session)

            response = StarletteStreamingResponse(
                control_body(),
                media_type="application/octet-stream",
                headers={RUST_RESPONSES_SESSION_HEADER: session.session_id},
            )
            setattr(response, "current_info", execution.current_info)
            return response
        return await execution.run()


class _ResponsesQueueBody:
    """Release each queue item only after its downstream send completes."""

    def __init__(
        self,
        execution: "ResponsesRequestExecution",
        worker_task: asyncio.Task[Any],
        first_lease: StreamQueueItemLease,
    ) -> None:
        self._execution = execution
        self._worker_task = worker_task
        self._current_lease: StreamQueueItemLease | None = first_lease
        self._first_pending = True
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    def __aiter__(self) -> "_ResponsesQueueBody":
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration

        if self._first_pending:
            self._first_pending = False
            assert self._current_lease is not None
            return self._current_lease.item

        if self._current_lease is not None:
            await self._current_lease.release()
            self._current_lease = None

        queue = self._execution.stream_output_queue
        assert queue is not None
        try:
            self._current_lease = await queue.get_lease()
        except StreamQueueClosed:
            await self.aclose()
            raise StopAsyncIteration from None
        except BaseException:
            # An upstream/local producer failure is not evidence of a client
            # disconnect.  Still release the in-flight item and unregister the
            # queue before propagating the precise error.
            await self.aclose()
            raise
        return self._current_lease.item

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_once())
        task = self._close_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            task.result()
            raise

    async def _close_once(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._current_lease is not None:
                await self._current_lease.release()
                self._current_lease = None
            await self._execution._close_stream_body(
                self._worker_task,
            )


async def _finish_stream_queue_cleanup_task(
    task: asyncio.Task[None],
) -> None:
    """Complete a queue ownership handoff before propagating cancellation."""

    pending_cancel: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            pending_cancel = pending_cancel or exc
    task.result()
    if pending_cancel is not None:
        raise pending_cancel


@dataclass(frozen=True, slots=True)
class CachedResponsesFailureTerminal:
    data: bytes
    status_code: int
    error_code: Optional[str]
    error_type: Optional[str]


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
    stream_output_queue: Optional[ByteBoundedQueue] = None
    stream_response_headers: dict[str, str] = field(default_factory=dict)
    stream_body_started: bool = False
    stream_keepalive_sent: bool = False
    stream_precommit_chunks: Optional[ReservedChunkBuffer] = None
    last_response_failed_terminal: Optional[CachedResponsesFailureTerminal] = None
    stream_phase_diagnostics: ResponsesStreamDiagnostics | None = field(
        default=None,
        repr=False,
    )

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
        _record_rust_request_spool_observability(http_request, current_info)
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_id = _responses_request_id(current_info)
        request_body_bytes = _request_body_size_bytes(http_request, request_data)
        request_type = detect_request_type(endpoint, request_data)
        if request_type:
            current_info["semantic_request_type"] = request_type
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            handler.last_provider_indices,
            handler.locks,
            endpoint=endpoint,
            request_body_bytes=request_body_bytes,
            request_type=request_type,
            reasoning_effort=request_reasoning_effort(request_data),
            debug=is_debug,
            provider_resolver=get_right_order_providers,
        )
        _record_plan_observability(current_info, plan)
        runner = UpstreamRunner(
            plan,
            endpoint=endpoint,
            debug=is_debug,
            clear_provider_auth_cache=lambda provider_api_key_raw: _codex_oauth_cache.pop(provider_api_key_raw, None),
            observability_context=current_info,
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
            try:
                response = await self._run_attempts()
            finally:
                await self._release_request_retry_payload()
        if isinstance(response, Response):
            setattr(response, "current_info", self.current_info)
        return response

    async def _release_request_retry_payload(self) -> None:
        if self.request_data is None:
            return
        # All retry-capable code has finished reading this model before this
        # handoff. Drop the object reference before returning its accounting.
        self.request_data = None
        request_scope = getattr(self.http_request, "scope", None)
        if not isinstance(request_scope, dict):
            return
        mark_request_body_releasable_at_response_start(request_scope)
        if not request_body_response_started(request_scope):
            return
        state = request_scope.get("state")
        lease = (
            state.get(ADMISSION_LEASE_STATE_KEY)
            if isinstance(state, dict)
            else None
        )
        if lease is None:
            lease = get_request_admission_lease()
        if lease is not None and lease.reserved_body_bytes:
            await lease.release_body_bytes(lease.reserved_body_bytes)

    async def _run_attempts(self, *, execute_attempt: Any = None):
        return await self.runner.run(
            execute_attempt or self._execute_attempt,
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
        self.stream_precommit_chunks = ReservedChunkBuffer(
            max_items=1,
            max_bytes=DEFAULT_MAX_EVENT_BYTES,
            retained_byte_budget=responses_stream_byte_budget,
        )
        self.stream_output_queue = ByteBoundedQueue(
            max_items=RESPONSES_STREAM_QUEUE_MAX_ITEMS,
            max_bytes=RESPONSES_STREAM_QUEUE_MAX_BYTES,
            put_timeout_seconds=RESPONSES_STREAM_QUEUE_PUT_TIMEOUT_SECONDS,
            retained_byte_budget=responses_stream_byte_budget,
        )
        runtime_gauges.register_stream_queue(self.stream_output_queue)
        worker_task = asyncio.create_task(self._stream_worker())
        first_lease: StreamQueueItemLease | None = None

        async def retire_unreturned_queue(
            lease_to_release: StreamQueueItemLease | None = None,
        ) -> None:
            try:
                if lease_to_release is not None:
                    await lease_to_release.release()
            finally:
                try:
                    await self.stream_output_queue.close(discard=True)
                finally:
                    try:
                        if not worker_task.done():
                            worker_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await worker_task
                    finally:
                        self._record_stream_queue_metrics()
                        runtime_gauges.unregister_stream_queue(
                            self.stream_output_queue
                        )

        async def finish_retiring_unreturned_queue(
            lease_to_release: StreamQueueItemLease | None = None,
        ) -> None:
            cleanup_task = asyncio.create_task(
                retire_unreturned_queue(lease_to_release)
            )
            await _finish_stream_queue_cleanup_task(cleanup_task)

        try:
            first_lease = await self.stream_output_queue.get_lease()
            first_item = first_lease.item
        except StreamQueueClosed:
            await finish_retiring_unreturned_queue()
            return Response(content="", status_code=204)
        except (
            StreamBufferBudgetTimeout,
            StreamQueuePutTimeout,
            StreamQueueItemTooLarge,
        ) as exc:
            self.current_info["status_code"] = 503
            self.current_info["stream_error_status_code"] = 503
            self.current_info["status_origin"] = "ember_local_admission"
            await finish_retiring_unreturned_queue(first_lease)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Local streaming buffer capacity exceeded",
                        "type": "local_overload",
                        "code": type(exc).__name__,
                    }
                },
                headers={
                    "retry-after": "1",
                    "x-uni-api-status-origin": "ember_local_admission",
                },
            )
        except BaseException:
            await finish_retiring_unreturned_queue(first_lease)
            raise
        if isinstance(first_item, Response):
            response_body = None
            response_memory_transferred = False
            try:
                self.current_info["status_code"] = first_item.status_code
                response_body = getattr(first_item, "body", b"")
                request_lease = get_request_admission_lease()
                if (
                    request_lease is not None
                    and isinstance(response_body, (bytes, bytearray, memoryview))
                    and response_body
                ):
                    await request_lease.reserve_response_bytes(len(response_body))
                response_memory_transferred = True
                response_body = None
                return first_item
            finally:
                response_body = None
                if not response_memory_transferred:
                    first_item = None
                await finish_retiring_unreturned_queue(first_lease)
        return StarletteStreamingResponse(
            _ResponsesQueueBody(self, worker_task, first_lease),
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
                    assert self.stream_precommit_chunks is not None
                    while self.stream_precommit_chunks:
                        chunk, reservation = self.stream_precommit_chunks.popleft()
                        if reservation is None:
                            await self._emit_stream_chunk(chunk)
                        else:
                            await self._emit_stream_chunk(
                                ReservedStreamChunk(chunk, reservation)
                            )
                    async with aclosing(response.body_iterator):
                        async for chunk in response.body_iterator:
                            await self._emit_stream_chunk(chunk)
                    return
                if not self.stream_body_started:
                    if self.stream_precommit_chunks is not None:
                        await self.stream_precommit_chunks.clear()
                    await self.stream_output_queue.put(response)
                    return
                if response.status_code != 499:
                    if self.last_response_failed_terminal is not None:
                        terminal = self.last_response_failed_terminal
                        await self._emit_stream_chunk(
                            ObservedStreamChunk(
                                terminal.data,
                                event_type="response.failed",
                                semantic_outcome="failed",
                                terminal_timeline_event=True,
                            )
                        )
                        diagnostics = self.current_info.get(
                            "_responses_stream_diagnostics_tracker"
                        )
                        if isinstance(
                            diagnostics,
                            ResponsesStreamDiagnostics,
                        ):
                            diagnostics.mark_terminal_queue_handoff_completed()
                        self.last_response_failed_terminal = None
                        self.current_info["stream_error_status_code"] = int(
                            terminal.status_code
                        )
                        self.current_info["stream_error_code"] = (
                            terminal.error_code
                        )
                        self.current_info["stream_error_type"] = (
                            terminal.error_type
                        )
                        self.current_info["stream_error_event_type"] = (
                            "response.failed"
                        )
                        self.current_info["stream_outcome"] = (
                            "upstream_failure_terminal"
                        )
                        self.current_info["success"] = False
                        return
                    await self._emit_stream_chunk(_stream_error_event_from_response(response))
                    await self._emit_stream_chunk(b"data: [DONE]\n\n")
            elif response is not None:
                await self._emit_stream_chunk(response)
        except asyncio.CancelledError:
            raise
        except (
            StreamBufferBudgetTimeout,
            StreamQueuePutTimeout,
            StreamQueueItemTooLarge,
        ) as exc:
            await self._abort_stream_for_backpressure(exc)
        except AdmissionRejected as exc:
            _record_local_admission_rejection(self.current_info, exc)
            await self._abort_stream_for_backpressure(exc)
        except Exception as exc:
            try:
                await self._handle_stream_worker_error(exc)
            except (
                StreamBufferBudgetTimeout,
                StreamQueuePutTimeout,
                StreamQueueItemTooLarge,
            ) as queue_exc:
                await self._abort_stream_for_backpressure(queue_exc)
        finally:
            await self._release_request_retry_payload()
            if self.stream_precommit_chunks is not None:
                await self.stream_precommit_chunks.clear()
            await self.stream_output_queue.close()

    async def _close_stream_body(
        self,
        worker_task: asyncio.Task[Any],
    ) -> None:
        assert self.stream_output_queue is not None
        # Closing a local iterator may mean a write deadline, server shutdown,
        # or another local failure.  Only the receive monitor is allowed to
        # assert a real peer disconnect on the shared sticky event.
        if not worker_task.done():
            diagnostics = self.current_info.get(
                "_responses_stream_diagnostics_tracker"
            )
            if isinstance(diagnostics, ResponsesStreamDiagnostics):
                if (
                    self.disconnect_event is not None
                    and self.disconnect_event.is_set()
                ):
                    diagnostics.mark_downstream_disconnect(
                        stage="downstream-body-iterator-close"
                    )
                    close_trigger = "downstream_disconnect"
                else:
                    diagnostics.mark_local_end(
                        origin="downstream_body_iterator_closed_or_shutdown"
                    )
                    close_trigger = "downstream_body_iterator_closed_or_shutdown"
                diagnostics.mark_cleanup_intent(
                    owner="responses_queue_body",
                    trigger=close_trigger,
                )
            worker_task.cancel()
        await self.stream_output_queue.close(discard=True)
        with suppress(asyncio.CancelledError):
            await worker_task
        self._record_stream_queue_metrics()
        runtime_gauges.unregister_stream_queue(self.stream_output_queue)

    async def _handle_stream_worker_error(self, exc: Exception) -> None:
        assert self.stream_output_queue is not None
        error_message = bounded_stream_error_text(exc)
        trace_logger.error(
            "%s stream worker failed request_id=%s model=%s error=%s",
            self.endpoint,
            self.request_id,
            self.request_model_name,
            error_message,
        )
        if not self.stream_body_started:
            if self.stream_precommit_chunks is not None:
                await self.stream_precommit_chunks.clear()
            self.current_info["status_code"] = 500
            self.current_info["stream_error_status_code"] = 500
            self.current_info["error_type"] = type(exc).__name__
            self.current_info["stream_outcome"] = "stream_worker_error"
            self.current_info["success"] = False
            await self.stream_output_queue.put(
                JSONResponse(status_code=500, content={"error": error_message})
            )
            return
        self.current_info["stream_error_status_code"] = 500
        self.current_info["error_type"] = type(exc).__name__
        self.current_info["stream_outcome"] = "stream_worker_error"
        self.current_info["success"] = False
        await self._emit_stream_chunk(
            _observed_responses_stream_error_event(500, error_message)
        )
        await self._emit_stream_chunk(b"data: [DONE]\n\n")

    async def _emit_stream_chunk(self, chunk: Any) -> None:
        if self.stream_output_queue is None:
            return
        diagnostics = self.stream_phase_diagnostics
        reservation = None
        event_type = None
        semantic_outcome = None
        terminal_timeline_event = False
        sse_metadata_complete = False
        usage_snapshot = None
        if isinstance(chunk, ReservedStreamChunk):
            reservation = chunk.reservation
            event_type = chunk.event_type
            semantic_outcome = chunk.semantic_outcome
            terminal_timeline_event = chunk.terminal_timeline_event
            sse_metadata_complete = chunk.sse_metadata_complete
            usage_snapshot = chunk.usage_snapshot
            chunk = chunk.data
        elif isinstance(chunk, ObservedStreamChunk):
            event_type = chunk.event_type
            semantic_outcome = chunk.semantic_outcome
            terminal_timeline_event = chunk.terminal_timeline_event
            sse_metadata_complete = chunk.sse_metadata_complete
            usage_snapshot = chunk.usage_snapshot
            chunk = chunk.data
        elif isinstance(chunk, ObservedStreamFrame):
            event_type = chunk.event_type
            semantic_outcome = chunk.semantic_outcome
            terminal_timeline_event = chunk.terminal_timeline_event
            sse_metadata_complete = chunk.sse_metadata_complete
            usage_snapshot = chunk.usage_snapshot
            chunk = chunk.data
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        if not isinstance(chunk, (bytes, bytearray)):
            chunk = str(chunk).encode("utf-8")
        self.stream_body_started = True
        chunk_bytes = bytes(chunk)
        segment_bytes = min(
            RESPONSES_STREAM_QUEUE_MAX_BYTES,
            256 * 1024,
        )
        try:
            for offset in range(0, len(chunk_bytes), segment_bytes):
                segment = chunk_bytes[offset : offset + segment_bytes]
                queue_item: Any = segment
                if (
                    event_type is not None
                    or semantic_outcome is not None
                    or sse_metadata_complete
                ):
                    queue_item = ObservedStreamChunk(
                        segment,
                        event_type=event_type,
                        semantic_outcome=semantic_outcome,
                        terminal_timeline_event=terminal_timeline_event,
                        final_event_segment=(
                            offset + len(segment) >= len(chunk_bytes)
                        ),
                        sse_metadata_complete=sse_metadata_complete,
                        usage_snapshot=usage_snapshot,
                    )
                transferred = (
                    reservation.split(len(segment))
                    if reservation is not None
                    else None
                )
                try:
                    queue_phase = (
                        diagnostics.begin_phase("queue_put")
                        if diagnostics is not None
                        else None
                    )
                    await self.stream_output_queue.put(
                        queue_item,
                        size=len(segment),
                        retained_byte_lease=transferred,
                    )
                except BaseException:
                    if transferred is not None and not transferred.released:
                        await transferred.release()
                    raise
                finally:
                    if diagnostics is not None:
                        diagnostics.finish_phase(
                            "queue_put",
                            queue_phase,
                            bytes_count=len(segment),
                            events=(
                                1
                                if offset + len(segment) >= len(chunk_bytes)
                                else 0
                            ),
                        )
            if not chunk_bytes:
                transferred = reservation.split(0) if reservation is not None else None
                try:
                    queue_phase = (
                        diagnostics.begin_phase("queue_put")
                        if diagnostics is not None
                        else None
                    )
                    await self.stream_output_queue.put(
                        b"",
                        retained_byte_lease=transferred,
                    )
                except BaseException:
                    if transferred is not None and not transferred.released:
                        await transferred.release()
                    raise
                finally:
                    if diagnostics is not None:
                        diagnostics.finish_phase(
                            "queue_put",
                            queue_phase,
                            events=1,
                        )
        finally:
            if reservation is not None and not reservation.released:
                await reservation.release()

    def _record_stream_queue_metrics(self) -> None:
        if self.stream_output_queue is None:
            return
        snapshot = self.stream_output_queue.snapshot()
        self.current_info["stream_queue_peak_items"] = snapshot.peak_items
        self.current_info["stream_queue_peak_bytes"] = snapshot.peak_bytes
        self.current_info["stream_queue_blocked_puts"] = snapshot.blocked_puts
        self.current_info["stream_queue_put_wait_ms"] = int(
            round(snapshot.put_wait_ms)
        )
        self.current_info["stream_queue_put_timeouts"] = snapshot.put_timeouts

    async def _abort_stream_for_backpressure(self, exc: Exception) -> None:
        assert self.stream_output_queue is not None
        diagnostics = self.current_info.get(
            "_responses_stream_diagnostics_tracker"
        )
        if isinstance(diagnostics, ResponsesStreamDiagnostics):
            diagnostics.mark_local_end(origin="local_backpressure_abort")
            diagnostics.mark_cleanup_intent(
                owner="responses_stream_queue",
                trigger="local_backpressure_abort",
            )
        _record_local_admission_rejection(self.current_info, exc)
        self.current_info["stream_error_status_code"] = 503
        self.current_info["error_type"] = type(exc).__name__
        self.current_info["stream_outcome"] = "local_backpressure_abort"
        self.current_info["success"] = False
        await self.stream_output_queue.close(error=exc, discard=True)

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
        assert self.stream_precommit_chunks is not None
        await self.stream_precommit_chunks.append(chunk_bytes)
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
            if responses_stream_stats_queue is not None:
                _enqueue_responses_stream_stats(args, kwargs)
                return
        _schedule_channel_stats_bounded(
            *args,
            **kwargs,
            fallback_background_tasks=self.background_tasks,
        )

    async def _before_next_attempt(self):
        if self.stream_output_queue is not None and not self.stream_body_started:
            if self.stream_precommit_chunks is not None:
                await self.stream_precommit_chunks.clear()
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
        precommit_semantic_guard = _responses_precommit_semantic_guard(
            engine,
        )
        attempt.state.update(
            {
                "upstream_url": upstream_url,
                "channel_id": channel_id,
                "engine": engine,
                "responses_stream_commit_policy": str(commit_policy or "real_output"),
                "responses_precommit_semantic_guard": precommit_semantic_guard,
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
            request_type=self.plan.request_type,
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
        attempt.state["failure_stage"] = "validation"
        payload = self._build_payload(attempt)
        json_payload = await run_json_cpu(json.dumps, payload)
        attempt.state["payload_bytes"] = len(json_payload.encode("utf-8"))
        self._record_upstream_attempt_start(attempt)
        self._log_attempt(attempt, headers, payload)

        async with app.state.client_manager.get_client(upstream_url, proxy, http2=False if engine == "codex" else None) as client:
            if self.request_data.stream:
                return await self._execute_stream_attempt(client, attempt, headers, json_payload)
            return await self._execute_non_stream_attempt(client, attempt, headers, json_payload)

    async def _build_rust_stream_plan(self, attempt: Any) -> dict[str, Any]:
        """Serialize one prepared retry attempt for the loopback Rust worker."""

        headers = self._build_headers(attempt)
        attempt.state["failure_stage"] = "validation"
        payload = self._build_payload(attempt)
        json_payload = await run_json_cpu(
            json.dumps,
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload_bytes = len(json_payload.encode("utf-8"))
        attempt.state["payload_bytes"] = payload_bytes
        attempt.state["failure_stage"] = "upstream"
        attempt.state["track_channel_stats"] = True
        self._record_upstream_attempt_start(attempt)
        self._log_attempt(attempt, headers, payload)
        _mark_current_info_stage(self.current_info, "upstream_send_start")
        runtime_gauges.begin_waiting_first_byte(self.current_info)
        attempt.state["rust_waiting_first_byte"] = True

        return {
            "kind": "plan",
            "protocol_version": 1,
            "session_id": str(
                self.current_info.get("rust_responses_session_id") or ""
            ),
            "attempt_id": str(attempt.routing_attempt_id),
            "url": str(attempt.state["upstream_url"]),
            "headers": headers,
            "body": json_payload,
            "proxy": attempt.state.get("proxy"),
            "engine": str(attempt.state.get("engine") or "gpt"),
            "precommit_semantic_guard": bool(
                attempt.state.get("responses_precommit_semantic_guard", True)
            ),
            "http1_only": attempt.state.get("engine") == "codex",
            "commit_policy": str(
                attempt.state.get("responses_stream_commit_policy")
                or "real_output"
            ),
            "normalize_custom_tool_call_ids": bool(
                self._custom_tool_call_id_normalization_enabled(attempt)
            ),
            "first_byte_timeout_seconds": attempt.state.get(
                "first_byte_timeout"
            ),
            "idle_timeout_seconds": attempt.state.get("idle_timeout"),
            "total_timeout_seconds": attempt.state.get("total_timeout"),
            "max_event_bytes": UNLIMITED_SSE_BYTES,
            "max_precommit_items": RESPONSES_STREAM_PRECOMMIT_MAX_ITEMS,
            "max_precommit_bytes": RESPONSES_STREAM_PRECOMMIT_MAX_BYTES,
        }

    def _finish_rust_first_byte_wait(self, attempt: Any) -> None:
        if not attempt.state.pop("rust_waiting_first_byte", False):
            return
        runtime_gauges.end_waiting_first_byte(self.current_info)

    async def _observe_rust_stream_commit(
        self,
        attempt: Any,
        observation: dict[str, Any],
    ) -> None:
        business_committed = bool(observation.get("business_committed", True))
        if business_committed and attempt.state.get("rust_stream_committed"):
            return
        attempt.state["rust_stream_response_started"] = True
        self.current_info["rust_responses_external_committed"] = True
        if business_committed:
            attempt.state["rust_stream_committed"] = True
        self._finish_rust_first_byte_wait(attempt)
        _mark_current_info_stage(self.current_info, "upstream_headers_received")
        _mark_first_byte_observed(self.current_info)
        upstream_status = int(observation.get("upstream_status_code") or 200)
        self.current_info["status_code"] = upstream_status
        attempt.state["routing_wire_status_code"] = upstream_status
        attempt.state["stream_upstream_status_code"] = upstream_status
        self.current_info["rust_responses_commit_reason"] = str(
            observation.get("commit_reason") or "stream_event"
        )[:80]
        self.current_info["rust_responses_precommit_events"] = max(
            0,
            int(observation.get("precommit_events") or 0),
        )
        self.current_info["rust_responses_precommit_bytes"] = max(
            0,
            int(observation.get("precommit_bytes") or 0),
        )
        if business_committed:
            await self._release_request_retry_payload()

    def _apply_rust_stream_report(self, outcome: dict[str, Any]) -> None:
        integer_fields = {
            "rust_responses_upstream_bytes": "upstream_bytes",
            "rust_responses_downstream_bytes": "downstream_bytes",
            "rust_responses_sse_events": "event_count",
            "rust_responses_delta_events": "delta_events",
            "rust_responses_normalized_events": "normalized_events",
        }
        for target, source in integer_fields.items():
            try:
                self.current_info[target] = max(0, int(outcome.get(source) or 0))
            except (TypeError, ValueError):
                self.current_info[target] = 0
        usage = outcome.get("usage")
        if isinstance(usage, dict):
            snapshot = stream_usage_snapshot(usage)
            if snapshot is not None and snapshot.parse_error is None:
                if snapshot.prompt_tokens is not None:
                    self.current_info["prompt_tokens"] = snapshot.prompt_tokens
                if snapshot.completion_tokens is not None:
                    self.current_info["completion_tokens"] = snapshot.completion_tokens
                if snapshot.total_tokens is not None:
                    self.current_info["total_tokens"] = snapshot.total_tokens
                if snapshot.complete:
                    self.current_info["usage_seen"] = True

    async def _resolve_rust_stream_outcome(
        self,
        attempt: Any,
        outcome: dict[str, Any],
    ) -> Any:
        kind = str(outcome.get("kind") or "protocol_error")
        upstream_status = int(outcome.get("upstream_status_code") or 200)
        attempt.state["routing_wire_status_code"] = upstream_status
        self._apply_rust_stream_report(outcome)

        if kind in {
            "http_error",
            "semantic_error",
            "transport_error",
            "protocol_error",
        } and not bool(outcome.get("committed")):
            self._finish_rust_first_byte_wait(attempt)
            if kind == "semantic_error":
                semantic = _responses_failure_http_exception(
                    outcome.get("payload"),
                    event_type=str(outcome.get("event_type") or "") or None,
                    wire_status_code=upstream_status,
                    validated_provider_sse=True,
                )
                if semantic is not None:
                    raise semantic
            status_code = int(outcome.get("status_code") or 502)
            detail = bounded_stream_error_text(
                outcome.get("detail")
                or outcome.get("body")
                or f"Rust upstream {kind}"
            )
            raise HTTPException(status_code=status_code, detail=detail)

        if not attempt.state.get("rust_stream_committed"):
            await self._observe_rust_stream_commit(
                attempt,
                {
                    "upstream_status_code": upstream_status,
                    "commit_reason": kind,
                    "precommit_events": outcome.get("precommit_events"),
                    "precommit_bytes": outcome.get("precommit_bytes"),
                },
            )

        if kind == "downstream_disconnected":
            self._finalize_stream_attempt_cancelled(
                attempt,
                downstream_disconnected=True,
            )
            return Response(content="", status_code=499)

        if kind in {"completed", "incomplete"}:
            self.current_info["stream_outcome"] = kind
            self._finalize_stream_attempt_success(attempt)
            response = Response(content="", status_code=upstream_status)
            setattr(
                response,
                "_uni_api_response_attempt_terminal_outcome",
                "stream_completed",
            )
            return response

        status_code = int(outcome.get("status_code") or 502)
        event_type = str(outcome.get("event_type") or "") or None
        error_code = str(outcome.get("error_code") or "") or None
        failure = SSEProtocolError(
            bounded_stream_error_text(
                outcome.get("detail") or f"Rust post-commit {kind}"
            )
        )
        terminal_bytes = b""
        if kind == "semantic_failure":
            semantic = _responses_failure_http_exception(
                outcome.get("payload"),
                event_type=event_type,
                wire_status_code=upstream_status,
                validated_provider_sse=True,
            )
            if semantic is not None:
                status_code = int(semantic.status_code)
                error_code = semantic.error_code
                event_type = semantic.responses_sse_event_type
                terminal_bytes = _encode_responses_sse_event(
                    semantic.responses_sse_event_type,
                    semantic.responses_sse_payload,
                )
                failure = SSEProtocolError(
                    "Responses upstream emitted a failure terminal: "
                    f"{semantic.detail_json[:1000]}"
                )
        self.current_info["stream_error_status_code"] = status_code
        self.current_info["stream_error_event_type"] = event_type
        self.current_info["stream_error_code"] = error_code
        self.current_info["stream_outcome"] = (
            "upstream_failure_terminal"
            if kind == "semantic_failure"
            else "upstream_stream_abort"
        )
        self.current_info["success"] = False
        self._finalize_stream_attempt_failure(
            attempt,
            failure,
            status_code=status_code,
            error_type=(
                "UpstreamSemanticFailure"
                if kind == "semantic_failure"
                else type(failure).__name__
            ),
            terminal_event_type=event_type,
            error_code=error_code,
        )
        response = Response(
            content=terminal_bytes,
            status_code=status_code,
            media_type="text/event-stream" if terminal_bytes else None,
        )
        return RustResponsesResolvedOutcome(
            runner_result=response,
            reply={
                "terminal_b64": (
                    base64.b64encode(terminal_bytes).decode("ascii")
                    if terminal_bytes
                    else ""
                )
            },
        )

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
            headers.setdefault("Session_id", self.http_request.headers.get("Session_id") or str(uuid.uuid4()))
            headers.setdefault("User-Agent", CODEX_USER_AGENT)
            headers.setdefault("Accept", "text/event-stream" if self.request_data.stream else "application/json")
            if codex_account_id:
                headers.setdefault("Chatgpt-Account-Id", str(codex_account_id))
        apply_provider_preference_headers(headers, attempt.provider, http_request=self.http_request)
        if engine == "codex":
            force_codex_client_headers(headers)
        _add_trace_headers(headers, self.current_info)
        apply_oaix_routing_attempt_id(
            headers,
            provider=attempt.provider,
            routing_attempt_id=attempt.routing_attempt_id,
        )
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
            skip_keys={"service_tier", "translation_options"},
        )
        if engine == "codex":
            strip_unsupported_codex_payload_fields(payload, strip_store=attempt.state["wants_compact"])
        if self._custom_tool_call_id_normalization_enabled(attempt):
            try:
                result = ResponsesCustomToolCallIdNormalizer().normalize(payload)
            except ResponsesCustomToolCallIdCollisionError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._record_custom_tool_call_id_normalization(
                attempt,
                result,
                direction="input",
            )
        return payload

    def _custom_tool_call_id_normalization_enabled(self, attempt: Any) -> bool:
        return responses_custom_tool_call_id_normalization_enabled(
            attempt.provider,
            (self.request_model_name, attempt.original_model),
        )

    def _record_custom_tool_call_id_normalization(
        self,
        attempt: Any,
        result: ResponsesCustomToolCallIdNormalizationResult,
        *,
        direction: str,
        event_type: Optional[str] = None,
    ) -> None:
        if not result.changed:
            return
        self.current_info["custom_tool_call_id_normalized"] = True
        self.current_info["responses_item_id_normalized"] = True
        self.current_info["custom_tool_call_id_normalization_count"] = int(
            self.current_info.get("custom_tool_call_id_normalization_count") or 0
        ) + int(result.normalized_ids)
        self.current_info["custom_tool_call_id_reference_rewrite_count"] = int(
            self.current_info.get("custom_tool_call_id_reference_rewrite_count") or 0
        ) + int(result.rewritten_references)
        self.current_info["responses_item_id_normalization_count"] = int(
            self.current_info.get("responses_item_id_normalization_count") or 0
        ) + int(result.normalized_ids)
        self.current_info["responses_item_id_reference_rewrite_count"] = int(
            self.current_info.get("responses_item_id_reference_rewrite_count") or 0
        ) + int(result.rewritten_references)
        trace_logger.info(
            "%s Responses item ID normalized request_id=%s provider=%s direction=%s event_type=%s normalized_ids=%s rewritten_references=%s paths=%s",
            self.endpoint,
            self.request_id,
            attempt.provider_name,
            direction,
            event_type or "none",
            result.normalized_ids,
            result.rewritten_references,
            ",".join(result.paths) or "none",
        )

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
            "routing_attempt_id": attempt.routing_attempt_id,
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
        terminal_event_type: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Any = None,
    ) -> None:
        finalize_routing_attempt(
            attempt,
            outcome="succeeded" if success else "failed",
            success=success,
            wire_status_code=status_code if success else None,
            semantic_status_code=None if success else status_code,
            terminal_event_type=terminal_event_type,
            error_code=error_code,
            error_type=error_type,
            error_message=error_message,
        )
        attempts = self.current_info.get("upstream_attempts")
        index = attempt.state.get("observability_attempt_index")
        if not isinstance(attempts, list) or not isinstance(index, int) or index < 0 or index >= len(attempts):
            return
        entry = attempts[index]
        if not isinstance(entry, dict):
            return
        entry["status_code"] = int(status_code)
        entry["success"] = bool(success)
        if attempt.state.get("local_admission_rejected"):
            entry["local_admission_rejected"] = True
        if attempt.state.get("status_origin"):
            entry["status_origin"] = str(attempt.state["status_origin"])[:64]
        routing_entry = attempt.state.get("_routing_attempt_entry")
        if isinstance(routing_entry, dict):
            for key in (
                "transport_error_kind",
                "transport_error_owner",
                "transport_error_phase",
                "transport_error_status_code",
                "provider_penalty_eligible",
                "local_overload",
                "failure_stage",
                "protocol_error_reason",
                "exception_type",
                "exception_module",
                "exception_repr",
                "exception_chain_json",
                "httpcore_exception_type",
                "httpcore_exception_module",
                "httpcore_exception_repr",
                "httpcore_exception_chain_json",
                "upstream_http_status_code",
                "http_version",
                "alpn_protocol",
                "http2_stream_id",
                "connection_request_count",
                "http2_concurrent_streams",
                "http2_max_concurrent_streams",
                "connection_local_state",
                "http2_local_connection_state",
                "http2_local_stream_state",
                "goaway_error_code",
                "goaway_error_code_name",
                "goaway_last_stream_id",
                "connection_snapshot_json",
                "httpcore_events_json",
                "transport_dns_started_ms",
                "transport_dns_completed_ms",
                "transport_dns_duration_ms",
                "transport_dns_status",
                "transport_dns_error_type",
                "transport_dns_address_count",
                "transport_tcp_started_ms",
                "transport_tcp_completed_ms",
                "transport_tcp_duration_ms",
                "transport_tcp_status",
                "transport_tcp_error_type",
                "transport_tcp_attempt_count",
                "transport_tcp_failed_attempt_count",
                "transport_tls_started_ms",
                "transport_tls_completed_ms",
                "transport_tls_duration_ms",
                "transport_tls_status",
                "transport_tls_error_type",
                "transport_request_headers_started_ms",
                "transport_request_headers_completed_ms",
                "transport_request_headers_duration_ms",
                "transport_request_headers_status",
                "transport_request_headers_error_type",
                "transport_request_body_started_ms",
                "transport_request_body_completed_ms",
                "transport_request_body_duration_ms",
                "transport_request_body_status",
                "transport_request_body_error_type",
                "transport_response_headers_started_ms",
                "transport_response_headers_completed_ms",
                "transport_response_headers_duration_ms",
                "transport_response_headers_status",
                "transport_response_headers_error_type",
                "transport_first_body_ms",
                "transport_first_body_bytes",
            ):
                if key in routing_entry:
                    entry[key] = routing_entry[key]
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
        if "_performance_phase_sampled" not in self.current_info:
            self.current_info["_performance_phase_sampled"] = (
                worker_runtime_observer.should_sample_phase_request(
                    "responses_stream"
                )
            )
        diagnostics = ResponsesStreamDiagnostics(
            current_info=self.current_info,
            attempt_index=attempt.state.get("observability_attempt_index"),
            logical_authority=_upstream_logical_authority(
                attempt.state.get("upstream_url")
            ),
            proxy_configured=bool(attempt.state.get("proxy")),
            sse_chunk_observer=worker_runtime_observer.record_sse_chunk,
            sse_event_observer=worker_runtime_observer.record_sse_event,
            terminal_hop_observer=worker_runtime_observer.record_terminal_hop,
            terminal_marker_missing_observer=(
                worker_runtime_observer.record_terminal_marker_missing
            ),
            phase_sampled=bool(
                self.current_info.get("_performance_phase_sampled")
            ),
            phase_sample_rate=worker_runtime_observer.phase_sample_rate,
            phase_observer=worker_runtime_observer.record_phase_sample,
            socket_unread_observer=(
                worker_runtime_observer.record_socket_unread_bytes
            ),
        )
        self.stream_phase_diagnostics = (
            diagnostics if diagnostics.phase_sampled else None
        )
        attempt.state["responses_stream_diagnostics_tracker"] = diagnostics
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
            "extensions": {"trace": diagnostics.httpcore_trace},
        }
        if attempt.state.get("upstream_timeout") is not None:
            stream_kwargs["timeout"] = attempt.state["upstream_timeout"]
        stream_cm = client.stream(**stream_kwargs)

        async def cleanup_cancelled_stream_enter(response: Any) -> None:
            diagnostics.capture_response(response)
            diagnostics.mark_local_end(origin="cancelled_upstream_header_wait")
            await _close_observed_responses_upstream_stream_safely(
                stream_cm,
                response,
                diagnostics,
                owner="responses_stream_enter_cleanup",
                trigger="cancelled_upstream_header_wait",
            )

        try:
            upstream_resp = await _await_first_byte_deadline(
                stream_cm.__aenter__(),
                timeout_seconds=first_byte_timeout,
                deadline=first_byte_deadline,
                total_timeout_seconds=total_timeout,
                total_deadline=total_deadline,
                cancel_result_cleanup=cleanup_cancelled_stream_enter,
                disconnect_event=self.disconnect_event,
            )
        except DownstreamDisconnectedDuringWait:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            diagnostics.mark_downstream_disconnect(stage="upstream-response-headers")
            diagnostics.mark_local_end(origin="downstream_disconnect")
            _log_responses_downstream_disconnect(
                self.endpoint,
                self.current_info,
                model_id=self.request_model_name,
                provider_name=attempt.provider_name,
                stage="upstream-response-headers",
            )
            return Response(content="", status_code=499)
        except BaseException as exc:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            diagnostics.observe_exception(exc, origin="upstream_response_headers")
            diagnostics.mark_local_end(origin="upstream_response_headers_error")
            raise
        _mark_current_info_stage(self.current_info, "upstream_headers_received")
        diagnostics.capture_response(upstream_resp)
        attempt.state["routing_wire_status_code"] = int(
            upstream_resp.status_code
        )
        if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            raw = await read_limited_response_body(upstream_resp)
            diagnostics.mark_local_end(origin="upstream_non_success_status")
            await _close_observed_responses_upstream_stream_safely(
                stream_cm,
                upstream_resp,
                diagnostics,
                owner="responses_precommit_cleanup",
                trigger="upstream_non_success_status",
            )
            raise HTTPException(status_code=upstream_resp.status_code, detail=raw.text())

        diagnostics.set_phase("precommit")
        if self.stream_output_queue is not None:
            self.stream_response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        upstream_iter = ObservedResponseByteIterator(
            upstream_resp.aiter_bytes(),
            diagnostics,
        )
        try:
            precommit_semantic_guard = bool(
                attempt.state.get("responses_precommit_semantic_guard", True)
            )
            precommit_keepalive_callback = None
            if self.stream_output_queue is not None and precommit_semantic_guard:
                precommit_keepalive_callback = functools.partial(
                    self._emit_precommit_keepalive,
                    passthrough=True,
                )
            async def cleanup_cancelled_precommit(result: Any) -> None:
                buffer, _committed = result
                await buffer.clear()

            buffered_chunks, stream_committed = await _await_first_byte_deadline(
                _prime_responses_upstream_stream(
                    upstream_iter,
                    upstream_status_code=upstream_resp.status_code,
                    disconnect_event=self.disconnect_event,
                    commit_policy=attempt.state.get("responses_stream_commit_policy", "real_output"),
                    precommit_semantic_guard=precommit_semantic_guard,
                    precommit_keepalive_callback=precommit_keepalive_callback,
                    retained_byte_budget=responses_stream_byte_budget,
                    diagnostics=diagnostics,
                ),
                timeout_seconds=first_byte_timeout,
                deadline=first_byte_deadline,
                total_timeout_seconds=total_timeout,
                total_deadline=total_deadline,
                satisfied=lambda: self.stream_body_started,
                cancel_result_cleanup=cleanup_cancelled_precommit,
                disconnect_event=self.disconnect_event,
            )
            _mark_first_byte_observed(self.current_info)
            diagnostics.set_phase("postcommit")
        except DownstreamDisconnectedDuringWait:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            diagnostics.mark_downstream_disconnect(stage="before-stream-commit")
            diagnostics.mark_local_end(origin="downstream_disconnect")
            await _close_observed_responses_upstream_stream_safely(
                stream_cm,
                upstream_resp,
                diagnostics,
                owner="responses_precommit_cleanup",
                trigger="downstream_disconnect",
            )
            _log_responses_downstream_disconnect(
                self.endpoint,
                self.current_info,
                model_id=self.request_model_name,
                provider_name=attempt.provider_name,
                stage="before-stream-commit",
            )
            return Response(content="", status_code=499)
        except (HTTPException, ResponsesSemanticError) as exc:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            semantic_failure_terminal = (
                diagnostics.facts.get("semantic_terminal_outcome") == "failed"
            )
            if semantic_failure_terminal:
                # This exception is the deliberate status mapping of a parsed
                # provider failure terminal, not a second transport or protocol
                # exception. Preserve its independent consistency facts.
                diagnostics.mark_local_end(
                    origin="precommit_semantic_failure_terminal"
                )
            else:
                diagnostics.observe_exception(
                    exc.__cause__
                    if isinstance(exc.__cause__, BaseException)
                    else exc,
                    origin="precommit_http_error",
                )
                diagnostics.mark_local_end(origin="precommit_http_error")
            await _close_observed_responses_upstream_stream_safely(
                stream_cm,
                upstream_resp,
                diagnostics,
                owner="responses_precommit_cleanup",
                trigger=(
                    "precommit_semantic_failure_terminal"
                    if semantic_failure_terminal
                    else "precommit_http_error"
                ),
            )
            raise
        except RESPONSES_STREAM_NETWORK_ERRORS as exc:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            diagnostics.observe_exception(exc, origin="precommit_network_or_deadline")
            diagnostics.mark_local_end(origin="precommit_network_error")
            await _close_observed_responses_upstream_stream_safely(
                stream_cm,
                upstream_resp,
                diagnostics,
                owner="responses_precommit_cleanup",
                trigger="after_upstream_read_failure",
            )
            raise
        except BaseException as exc:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            diagnostics.observe_exception(exc, origin="precommit_local_error")
            diagnostics.mark_local_end(origin="precommit_local_error")
            await _close_observed_responses_upstream_stream_safely(
                stream_cm,
                upstream_resp,
                diagnostics,
                owner="responses_precommit_cleanup",
                trigger="precommit_local_error",
            )
            raise

        if self.disconnect_event is not None and self.disconnect_event.is_set():
            await buffered_chunks.clear()
            diagnostics.mark_downstream_disconnect(stage="before-stream-commit")
            diagnostics.mark_local_end(origin="downstream_disconnect")
            await _close_observed_responses_upstream_stream_safely(
                stream_cm,
                upstream_resp,
                diagnostics,
                owner="responses_precommit_cleanup",
                trigger="downstream_disconnect",
            )
            _log_responses_downstream_disconnect(
                self.endpoint,
                self.current_info,
                model_id=self.request_model_name,
                provider_name=attempt.provider_name,
                stage="before-stream-commit",
            )
            return Response(content="", status_code=499)

        attempt.state["stream_upstream_status_code"] = upstream_resp.status_code
        response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        await self._release_request_retry_payload()
        return StarletteStreamingResponse(
            self._proxy_responses_stream(
                attempt,
                buffered_chunks,
                upstream_iter,
                stream_cm,
                upstream_resp,
                stream_committed,
                diagnostics,
                total_deadline=total_deadline,
                total_timeout_seconds=total_timeout,
            ),
            media_type="text/event-stream",
            headers=response_headers,
        )

    async def _proxy_responses_stream(
        self,
        attempt: Any,
        buffered_chunks: ReservedChunkBuffer,
        upstream_iter: Any,
        stream_cm: Any,
        upstream_resp: Any,
        stream_committed: bool,
        diagnostics: ResponsesStreamDiagnostics,
        *,
        total_deadline: Optional[float] = None,
        total_timeout_seconds: Any = None,
    ):
        diagnostics.set_phase("postcommit")
        phase_diagnostics = diagnostics if diagnostics.phase_sampled else None
        completed_seen = False
        incomplete_seen = False
        usage_seen = False
        output_seen = False
        terminal_queue_handoff_completed = False
        fast_path_candidates = 0
        fast_path_events = 0
        fast_path_fallbacks = 0
        fast_path_bytes = 0
        # Responses events have no fixed frame/feed ceiling. The parser still
        # charges retained bytes and JSON work to the process/request memory
        # governors before materializing them.
        proxy_sse_parser = IncrementalSSEParser(
            max_pending_bytes=UNLIMITED_SSE_BYTES,
            max_event_bytes=UNLIMITED_SSE_BYTES,
            max_feed_bytes=UNLIMITED_SSE_BYTES,
        )
        custom_tool_call_id_normalizer = (
            ResponsesCustomToolCallIdNormalizer()
            if self._custom_tool_call_id_normalization_enabled(attempt)
            else None
        )
        item_id_requires_full_normalization = (
            custom_tool_call_id_normalizer.requires_item_id_full_normalization
            if custom_tool_call_id_normalizer is not None
            else None
        )
        parse_workspace = await ReusableJSONParseWorkspace.create()

        async def source_events():
            """Frame transport chunks before exposing any protocol event."""

            while buffered_chunks:
                chunk, reservation = buffered_chunks.popleft()
                raw_events = []
                try:
                    raw_events = proxy_sse_parser.feed(chunk)
                    # Precommit stores each complete canonical frame as its own
                    # bytes chunk.  If feeding that chunk yields exactly one
                    # event and leaves no parser tail, the original chunk is
                    # already the event wire.  Re-encoding the parsed text only
                    # to prove equality would violate the encode-once path.
                    single_event_wire = (
                        chunk
                        if len(raw_events) == 1
                        and not proxy_sse_parser.pending_data
                        else None
                    )
                    exact_transfer = (
                        reservation is not None
                        and single_event_wire is not None
                    )
                    for event_index in range(len(raw_events)):
                        raw_event = raw_events[event_index]
                        transferred = None
                        upstream_event_bytes = (
                            single_event_wire
                            if event_index == 0
                            else None
                        )
                        if exact_transfer and event_index == 0:
                            transferred = reservation
                            reservation = None
                        try:
                            yield (
                                raw_event,
                                transferred,
                                True,
                                True,
                                None,
                                upstream_event_bytes,
                            )
                        finally:
                            transferred = None
                            upstream_event_bytes = None
                            raw_event = None
                            raw_events[event_index] = ""
                finally:
                    raw_events.clear()
                    raw_events = None
                    chunk = None
                    if reservation is not None:
                        await reservation.release()

            while True:
                if self._downstream_disconnected(
                    attempt,
                    stage="after-stream-commit",
                ):
                    raise DownstreamDisconnectedDuringWait()
                receive_phase = (
                    phase_diagnostics.begin_phase("socket_receive")
                    if phase_diagnostics is not None
                    else None
                )
                try:
                    chunk = await _await_first_byte_deadline(
                        upstream_iter.__anext__(),
                        deadline=total_deadline,
                        total_timeout_seconds=total_timeout_seconds,
                        disconnect_event=self.disconnect_event,
                    )
                    chunk_receive_timing = _capture_stream_receive_timing()
                except StopAsyncIteration:
                    if phase_diagnostics is not None:
                        phase_diagnostics.finish_phase(
                            "socket_receive",
                            receive_phase,
                        )
                    break
                except BaseException:
                    if phase_diagnostics is not None:
                        phase_diagnostics.finish_phase(
                            "socket_receive",
                            receive_phase,
                        )
                    raise
                if phase_diagnostics is not None:
                    phase_diagnostics.finish_phase(
                        "socket_receive",
                        receive_phase,
                        bytes_count=len(chunk),
                    )
                if self._downstream_disconnected(
                    attempt,
                    stage="after-stream-commit",
                ):
                    # The disconnect may become visible while the upstream
                    # __anext__ call is completing.  Never forward the chunk
                    # that raced with that disconnect.
                    chunk = None
                    raise DownstreamDisconnectedDuringWait()
                chunk_size = len(chunk)
                chunk_bytes = bytes(chunk)
                frame_phase = (
                    phase_diagnostics.begin_phase("sse_frame")
                    if phase_diagnostics is not None
                    else None
                )
                raw_events = None
                fast_frame = None
                try:
                    if (
                        RESPONSES_DELTA_FAST_PATH_ENABLED
                        and output_seen
                        and chunk_size
                        >= MIN_CANONICAL_RESPONSES_DELTA_FRAME_BYTES
                        and chunk_bytes.startswith(b"event: response.")
                        and chunk_bytes.endswith(b"\n\n")
                        and proxy_sse_parser.can_bypass_complete_frame
                    ):
                        fast_frame = match_canonical_responses_delta_frame(
                            chunk_bytes,
                            max_event_bytes=UNLIMITED_SSE_BYTES,
                        )
                    if fast_frame is None:
                        raw_events = proxy_sse_parser.feed(chunk_bytes)
                finally:
                    if phase_diagnostics is not None:
                        phase_diagnostics.finish_phase(
                            "sse_frame",
                            frame_phase,
                            bytes_count=chunk_size,
                            events=(
                                1
                                if fast_frame is not None
                                else len(raw_events)
                                if isinstance(raw_events, list)
                                else 0
                            ),
                        )
                chunk = None
                if fast_frame is not None:
                    fast_frame_lease = StreamParserRetainedLease()
                    fast_frame_lease.grow(len(chunk_bytes) - 2)
                    try:
                        yield (
                            fast_frame,
                            None,
                            False,
                            False,
                            chunk_receive_timing,
                            chunk_bytes,
                        )
                    finally:
                        fast_frame_lease.release()
                        fast_frame_lease = None
                        fast_frame = None
                        chunk_bytes = None
                        chunk_receive_timing = None
                    continue
                assert isinstance(raw_events, list)
                chunk_bytes = None
                try:
                    for event_index in range(len(raw_events)):
                        raw_event = raw_events[event_index]
                        try:
                            yield (
                                raw_event,
                                None,
                                False,
                                False,
                                chunk_receive_timing,
                                None,
                            )
                        finally:
                            raw_event = None
                            raw_events[event_index] = ""
                finally:
                    raw_events.clear()
                    raw_events = None
                    chunk_receive_timing = None

            diagnostics.observe_partial_diagnostics(
                proxy_sse_parser.pending_diagnostics()
            )
            eof_receive_timing = _capture_stream_receive_timing()
            frame_phase = (
                phase_diagnostics.begin_phase("sse_frame")
                if phase_diagnostics is not None
                else None
            )
            try:
                raw_events = proxy_sse_parser.finish()
            finally:
                if phase_diagnostics is not None:
                    phase_diagnostics.finish_phase(
                        "sse_frame",
                        frame_phase,
                        events=(
                            len(raw_events)
                            if isinstance(raw_events, list)
                            else 0
                        ),
                    )
            try:
                for event_index in range(len(raw_events)):
                    raw_event = raw_events[event_index]
                    try:
                        yield (
                            raw_event,
                            None,
                            False,
                            False,
                            eof_receive_timing,
                            None,
                        )
                    finally:
                        raw_event = None
                        raw_events[event_index] = ""
            finally:
                raw_events.clear()
                raw_events = None
                eof_receive_timing = None

        source = source_events()
        try:
            async for (
                raw_event,
                reservation,
                from_precommit_buffer,
                already_observed,
                receive_timing,
                upstream_event_bytes,
            ) in source:
                event_bytes = None
                transferred = None
                failure = None
                event_owner = None
                usage_snapshot = None
                fast_frame = (
                    raw_event
                    if isinstance(raw_event, CanonicalResponsesDeltaFrame)
                    else None
                )
                fast_transparent = False
                received_at = (
                    receive_timing.received_at
                    if receive_timing is not None
                    else None
                )
                received_unix_nano = (
                    receive_timing.unix_nano
                    if receive_timing is not None
                    else None
                )
                received_monotonic_nano = (
                    receive_timing.monotonic_nano
                    if receive_timing is not None
                    else None
                )
                try:
                    if fast_frame is None and not raw_event.strip():
                        continue
                    if fast_frame is None and is_sse_comment_frame(raw_event):
                        if is_oaix_terminal_flush_marker(raw_event):
                            if not already_observed:
                                diagnostics.observe_complete_event(
                                    raw_event,
                                    has_data_field=False,
                                    received_at=received_at,
                                    received_unix_nano=received_unix_nano,
                                    received_monotonic_nano=(
                                        received_monotonic_nano
                                    ),
                                )
                            # This hop-local marker is consumed by diagnostics
                            # and must never change the downstream SSE contract.
                            continue
                        if upstream_event_bytes is None:
                            upstream_event_bytes = (
                                _raw_responses_sse_event_bytes(raw_event)
                            )
                        event_bytes = upstream_event_bytes
                        if not already_observed:
                            observer_phase = (
                                phase_diagnostics.begin_phase("observer_hash")
                                if phase_diagnostics is not None
                                else None
                            )
                            try:
                                diagnostics.observe_complete_event(
                                    raw_event,
                                    has_data_field=False,
                                    wire_bytes=event_bytes,
                                    received_at=received_at,
                                    received_unix_nano=received_unix_nano,
                                    received_monotonic_nano=(
                                        received_monotonic_nano
                                    ),
                                )
                            finally:
                                if phase_diagnostics is not None:
                                    phase_diagnostics.finish_phase(
                                        "observer_hash",
                                        observer_phase,
                                        bytes_count=len(event_bytes),
                                        events=1,
                                    )
                        if reservation is None:
                            yield event_bytes
                        else:
                            transferred = reservation
                            reservation = None
                            yield ReservedStreamChunk(event_bytes, transferred)
                        continue

                    parse_phase = (
                        phase_diagnostics.begin_phase("json_parse")
                        if phase_diagnostics is not None
                        else None
                    )
                    try:
                        if fast_frame is not None:
                            fast_path_candidates += 1
                            fast_transparent = await (
                                can_forward_responses_delta_without_materializing(
                                    fast_frame,
                                    workspace=parse_workspace,
                                    max_json_estimated_bytes=(
                                        RESPONSES_SSE_JSON_MAX_ESTIMATED_BYTES
                                    ),
                                    item_id_requires_full_normalization=(
                                        item_id_requires_full_normalization
                                    ),
                                )
                            )
                            if not fast_transparent:
                                fast_path_fallbacks += 1
                                raw_event = fast_frame.decode_raw_event()
                        if not fast_transparent:
                            event_owner = await parse_owned_sse_event(
                                raw_event,
                                max_event_bytes=UNLIMITED_SSE_BYTES,
                                max_json_estimated_bytes=(
                                    RESPONSES_SSE_JSON_MAX_ESTIMATED_BYTES
                                ),
                                workspace=parse_workspace,
                            )
                    except BaseException:
                        if upstream_event_bytes is None:
                            upstream_event_bytes = (
                                fast_frame.wire_bytes
                                if fast_frame is not None
                                else _raw_responses_sse_event_bytes(raw_event)
                            )
                        if not already_observed:
                            observer_phase = (
                                phase_diagnostics.begin_phase("observer_hash")
                                if phase_diagnostics is not None
                                else None
                            )
                            try:
                                diagnostics.observe_complete_event(
                                    (
                                        raw_event
                                        if isinstance(raw_event, str)
                                        else ""
                                    ),
                                    has_data_field=(
                                        True
                                        if fast_frame is not None
                                        else sse_event_has_data_field(raw_event)
                                    ),
                                    event_type=(
                                        fast_frame.event_type
                                        if fast_frame is not None
                                        else None
                                    ),
                                    wire_bytes=upstream_event_bytes,
                                    received_at=received_at,
                                    received_unix_nano=received_unix_nano,
                                    received_monotonic_nano=(
                                        received_monotonic_nano
                                    ),
                                    force_hash=True,
                                )
                            finally:
                                if phase_diagnostics is not None:
                                    phase_diagnostics.finish_phase(
                                        "observer_hash",
                                        observer_phase,
                                        bytes_count=len(upstream_event_bytes),
                                        events=1,
                                    )
                        raise
                    finally:
                        if phase_diagnostics is not None:
                            phase_diagnostics.finish_phase(
                                "json_parse",
                                parse_phase,
                                bytes_count=(
                                    len(upstream_event_bytes)
                                    if upstream_event_bytes is not None
                                    else 0
                                ),
                                events=1,
                            )
                    if fast_transparent:
                        assert fast_frame is not None
                        assert upstream_event_bytes is not None
                        event_bytes = upstream_event_bytes
                        observer_phase = (
                            phase_diagnostics.begin_phase("observer_hash")
                            if phase_diagnostics is not None
                            else None
                        )
                        try:
                            diagnostics.observe_complete_event(
                                "",
                                has_data_field=True,
                                event_type=fast_frame.event_type,
                                wire_bytes=upstream_event_bytes,
                                received_at=received_at,
                                received_unix_nano=received_unix_nano,
                                received_monotonic_nano=(
                                    received_monotonic_nano
                                ),
                            )
                        finally:
                            if phase_diagnostics is not None:
                                phase_diagnostics.finish_phase(
                                    "observer_hash",
                                    observer_phase,
                                    bytes_count=len(upstream_event_bytes),
                                    events=1,
                                )
                        fast_path_events += 1
                        fast_path_bytes += len(event_bytes)
                        yield _observed_responses_stream_chunk(
                            event_bytes,
                            None,
                            event_type=fast_frame.event_type,
                            semantic_outcome="nonterminal",
                        )
                        continue
                    event_type = event_owner.event_name
                    event_payload = event_owner.payload
                    diagnostics.mark_terminal_parse_completed(
                        event_type,
                        received_at=received_at,
                        received_unix_nano=received_unix_nano,
                        received_monotonic_nano=received_monotonic_nano,
                    )
                    if upstream_event_bytes is None:
                        upstream_event_bytes = _raw_responses_sse_event_bytes(
                            raw_event
                        )
                    if not already_observed:
                        observer_phase = (
                            phase_diagnostics.begin_phase("observer_hash")
                            if phase_diagnostics is not None
                            else None
                        )
                        try:
                            diagnostics.observe_complete_event(
                                raw_event,
                                has_data_field=event_owner.has_data_field,
                                event_type=event_type,
                                wire_bytes=upstream_event_bytes,
                                received_at=received_at,
                                received_unix_nano=received_unix_nano,
                                received_monotonic_nano=(
                                    received_monotonic_nano
                                ),
                            )
                        finally:
                            if phase_diagnostics is not None:
                                phase_diagnostics.finish_phase(
                                    "observer_hash",
                                    observer_phase,
                                    bytes_count=len(upstream_event_bytes),
                                    events=1,
                                )
                    semantic_failure = None
                    terminal_success = False
                    try:
                        if not event_owner.has_data_field:
                            diagnostics.observe_normalization(
                                "ignored_no_data_event_block",
                                event_owner.declared_event_name,
                            )
                            continue

                        if event_type == "[DONE]":
                            raise SSEProtocolError(
                                "Responses upstream emitted [DONE] without a terminal response event"
                            )

                        validate_sse_event_type_consistency(
                            event_owner.declared_event_name,
                            event_payload,
                            protocol_name="Responses",
                            has_event_field=event_owner.has_event_field,
                            require_event_name=True,
                        )
                        event_bytes, event_was_normalized = (
                            _canonical_responses_sse_event_bytes(
                                raw_event,
                                event_type=event_type,
                                has_event_field=event_owner.has_event_field,
                                wire_bytes=upstream_event_bytes,
                            )
                        )
                        if (
                            not event_owner.has_event_field
                            and event_type
                            and event_type != "[DONE]"
                        ):
                            diagnostics.observe_normalization(
                                "canonicalized_data_only_event",
                                event_type,
                            )
                        if event_was_normalized and reservation is not None:
                            await reservation.release()
                            reservation = None

                        if custom_tool_call_id_normalizer is not None:
                            try:
                                normalization_result = (
                                    custom_tool_call_id_normalizer.normalize(
                                        event_payload
                                    )
                                )
                            except ResponsesCustomToolCallIdCollisionError as exc:
                                raise SSEProtocolError(str(exc)) from exc
                            if normalization_result.changed:
                                event_bytes = _encode_responses_sse_event(
                                    event_type,
                                    event_payload,
                                )
                                if reservation is not None:
                                    await reservation.release()
                                    reservation = None
                                diagnostics.observe_normalization(
                                    "custom_tool_call_id_prefix",
                                    event_type,
                                )
                                self._record_custom_tool_call_id_normalization(
                                    attempt,
                                    normalization_result,
                                    direction="output",
                                    event_type=event_type,
                                )

                        _validate_responses_terminal_payload(
                            event_type,
                            event_payload,
                        )
                        if (
                            not output_seen
                            and _responses_stream_event_has_real_output(
                                event_type,
                                event_payload,
                            )
                        ):
                            output_seen = True

                        semantic_failure = _responses_failure_http_exception(
                            event_payload,
                            event_type=event_type,
                            wire_status_code=upstream_resp.status_code,
                            validated_provider_sse=True,
                        )
                        if semantic_failure is not None:
                            semantic_outcome = "failed"
                        elif event_type == "response.completed":
                            semantic_outcome = "completed"
                        elif event_type == "response.incomplete":
                            semantic_outcome = "incomplete"
                        else:
                            semantic_outcome = "nonterminal"
                        observer_phase = (
                            phase_diagnostics.begin_phase("observer_hash")
                            if phase_diagnostics is not None
                            else None
                        )
                        try:
                            diagnostics.observe_parsed_event(
                                raw_event,
                                event_type,
                                event_payload,
                                semantic_outcome=semantic_outcome,
                                wire_bytes=upstream_event_bytes,
                                received_at=received_at,
                            )
                        finally:
                            if phase_diagnostics is not None:
                                phase_diagnostics.finish_phase(
                                    "observer_hash",
                                    observer_phase,
                                )
                        # The JSON graph is already materialized here.  Read
                        # the bounded scalar metadata directly instead of
                        # relying on a textual key prefilter: JSON object keys
                        # may legally contain escapes (for example,
                        # ``"\\u0075sage"``).
                        usage_snapshot = stream_usage_snapshot_from_payload(
                            event_payload
                        )
                        if semantic_failure is not None:
                            downstream_event_type = event_type
                            if (
                                event_type == "error"
                                and semantic_failure.responses_sse_event_type
                                == "response.failed"
                            ):
                                normalized_event_bytes = (
                                    _encode_responses_sse_event(
                                        "response.failed",
                                        semantic_failure.responses_sse_payload,
                                    )
                                )
                                event_bytes = normalized_event_bytes
                                normalized_event_bytes = None
                                if reservation is not None:
                                    raw_reservation = reservation
                                    reservation = None
                                    await raw_reservation.release()
                                    raw_reservation = None
                                downstream_event_type = "response.failed"
                                diagnostics.observe_normalization(
                                    "provider_error_to_response_failed",
                                    event_type,
                                )
                            failure = SSEProtocolError(
                                "Responses upstream emitted a failure terminal: "
                                f"{semantic_failure.detail_json[:1000]}"
                            )
                            self.current_info["stream_error_status_code"] = int(
                                semantic_failure.status_code
                            )
                            self.current_info["stream_outcome"] = (
                                "upstream_failure_terminal"
                            )
                            self.current_info["error_type"] = "UpstreamSemanticFailure"
                            self.current_info["stream_error_code"] = (
                                semantic_failure.error_code
                            )
                            self.current_info["stream_error_type"] = (
                                semantic_failure.error_type
                            )
                            self.current_info["stream_error_event_type"] = (
                                downstream_event_type
                            )
                            self.current_info["success"] = False
                            if semantic_failure.status_code in (400, 413):
                                attempt.state["track_channel_stats"] = False
                            self._finalize_stream_attempt_failure(
                                attempt,
                                failure,
                                status_code=int(semantic_failure.status_code),
                                error_type=(
                                    semantic_failure.error_type
                                    or "UpstreamSemanticFailure"
                                ),
                                terminal_event_type=event_type,
                                error_code=semantic_failure.error_code,
                            )
                            # Preserve the downstream-compatible structured terminal.
                            if reservation is not None:
                                transferred = reservation
                                reservation = None
                            yield _observed_responses_stream_chunk(
                                event_bytes,
                                transferred,
                                event_type=downstream_event_type,
                                semantic_outcome="failed",
                                terminal_timeline_event=True,
                                usage_snapshot=usage_snapshot,
                            )
                            terminal_queue_handoff_completed = True
                            diagnostics.mark_terminal_queue_handoff_completed()
                            diagnostics.mark_local_end(
                                origin="upstream_failure_terminal"
                            )
                            await event_owner.aclose()
                            event_owner = None
                            event_payload = None
                            await _consume_expected_oaix_terminal_flush_marker(
                                source,
                                diagnostics,
                            )
                            return

                        terminal_success = event_type in {
                            "response.completed",
                            "response.incomplete",
                        }
                        if event_type == "response.completed":
                            completed_seen = True
                        elif event_type == "response.incomplete":
                            incomplete_seen = True
                        if terminal_success:
                            usage_seen = bool(diagnostics.facts.get("usage_seen"))

                        if terminal_success:
                            if reservation is not None:
                                transferred = reservation
                                reservation = None
                            yield _observed_responses_stream_chunk(
                                event_bytes,
                                transferred,
                                event_type=event_type,
                                semantic_outcome=semantic_outcome,
                                terminal_timeline_event=True,
                                usage_snapshot=usage_snapshot,
                            )
                        else:
                            if reservation is not None:
                                transferred = reservation
                                reservation = None
                            yield _observed_responses_stream_chunk(
                                event_bytes,
                                transferred,
                                event_type=event_type,
                                semantic_outcome=semantic_outcome,
                                terminal_timeline_event=False,
                                usage_snapshot=usage_snapshot,
                            )

                        if terminal_success:
                            # The semantic terminal ends the business response.
                            # Do not wait for EOF/[DONE] on a keep-alive
                            # connection. When OAIX explicitly advertises its
                            # marker contract, consume only that immediate,
                            # bounded diagnostic comment.
                            self._finalize_stream_attempt_success(attempt)
                            terminal_queue_handoff_completed = True
                            diagnostics.mark_terminal_queue_handoff_completed()
                            diagnostics.mark_local_end(
                                origin=f"semantic_{event_type}"
                            )
                            await event_owner.aclose()
                            event_owner = None
                            event_payload = None
                            await _consume_expected_oaix_terminal_flush_marker(
                                source,
                                diagnostics,
                            )
                            return
                    finally:
                        terminal_success = False
                        semantic_failure = None
                        event_payload = None
                        event_type = None
                        if event_owner is not None:
                            await event_owner.aclose()
                finally:
                    failure = None
                    transferred = None
                    event_bytes = None
                    upstream_event_bytes = None
                    raw_event = None
                    event_owner = None
                    usage_snapshot = None
                    from_precommit_buffer = False
                    already_observed = False
                    received_at = None
                    received_unix_nano = None
                    received_monotonic_nano = None
                    receive_timing = None
                    if reservation is not None:
                        await reservation.release()
            raise SSEProtocolError(
                "Responses upstream ended without a terminal response event"
            )
        except DownstreamDisconnectedDuringWait:
            diagnostics.mark_downstream_disconnect(stage="after-stream-commit")
            diagnostics.mark_local_end(origin="downstream_disconnect")
            self._finalize_stream_attempt_cancelled(
                attempt,
                downstream_disconnected=True,
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Starlette BaseHTTPMiddleware transports inner response body
            # messages over an anyio stream and may re-raise an inner iterator
            # exception only after the outer response has observed a clean EOF.
            # Handle every ordinary post-commit generator failure here, before
            # that boundary, so it is recorded as a failed upstream attempt.
            #
            # Do not turn a transport/protocol truncation into a synthetic
            # Responses semantic terminal.  Codex clients distinguish an
            # incomplete stream (no response terminal and no [DONE]) from an
            # explicit provider error and may safely retry only the former.
            # Returning normally after the already-forwarded prefix preserves
            # that distinction while still containing the iterator exception
            # inside this ASGI body owner.
            if proxy_sse_parser.failure_pending_diagnostics is not None:
                diagnostics.observe_partial_diagnostics(
                    proxy_sse_parser.failure_pending_diagnostics
                )
            else:
                diagnostics.observe_partial_diagnostics(
                    proxy_sse_parser.pending_diagnostics()
                )
            diagnostics.observe_exception(exc, origin="postcommit_stream")
            diagnostics.mark_local_end(origin="postcommit_stream_error")
            self._finalize_stream_attempt_failure(attempt, exc)
            await self._handle_proxy_stream_abort(attempt, exc, stream_committed)
            if stream_committed:
                diagnostics.mark_postcommit_stream_terminal_suppressed()
                self.current_info["stream_error_status_code"] = 502
                self.current_info["error_type"] = type(exc).__name__
                self.current_info["stream_outcome"] = "upstream_stream_abort"
                self.current_info["success"] = False
                self.current_info["stream_error_after_response_start"] = True
                self.current_info["postcommit_stream_terminal_suppressed"] = True
                return
        finally:
            if fast_path_candidates:
                self.current_info["responses_delta_fast_path_candidates"] = (
                    fast_path_candidates
                )
                self.current_info["responses_delta_fast_path_events"] = (
                    fast_path_events
                )
                self.current_info["responses_delta_fast_path_fallbacks"] = (
                    fast_path_fallbacks
                )
                self.current_info["responses_delta_fast_path_bytes"] = (
                    fast_path_bytes
                )
            try:
                try:
                    await source.aclose()
                finally:
                    await parse_workspace.aclose()
            finally:
                pending_diagnostics = (
                    proxy_sse_parser.failure_pending_diagnostics
                )
                diagnostics.observe_partial_diagnostics(
                    pending_diagnostics
                    if pending_diagnostics is not None
                    else proxy_sse_parser.pending_diagnostics()
                )
                proxy_sse_parser.discard()
                try:
                    await buffered_chunks.clear()
                finally:
                    try:
                        if not attempt.state.get("stream_attempt_finalized"):
                            if completed_seen or incomplete_seen:
                                self._finalize_stream_attempt_success(attempt)
                            elif (
                                self.disconnect_event is not None
                                and self.disconnect_event.is_set()
                            ):
                                self._finalize_stream_attempt_cancelled(
                                    attempt,
                                    downstream_disconnected=True,
                                )
                            else:
                                # A consumer may close an iterator for server
                                # shutdown or another local reason.  Do not
                                # fabricate a peer disconnect/provider fault.
                                attempt.state["stream_attempt_finalized"] = True
                                finalize_routing_attempt(
                                    attempt,
                                    outcome="consumer_or_shutdown_unknown",
                                    success=None,
                                )
                                finalize_response_memory_attempt(
                                    attempt=attempt,
                                    outcome="consumer_or_shutdown_unknown",
                                )
                        terminal_or_usage_missing = (
                            not (completed_seen or incomplete_seen)
                            or not usage_seen
                        )
                        if (
                            terminal_or_usage_missing
                            and diagnostics.facts.get("downstream_disconnected")
                        ):
                            trace_logger.info(
                                "%s upstream read cancelled after downstream disconnect before completed usage request_id=%s model=%s provider=%s output_seen=%s completed_seen=%s usage_seen=%s upstream_url=%s",
                                self.endpoint,
                                self.request_id,
                                self.request_model_name,
                                attempt.provider_name,
                                output_seen,
                                completed_seen or incomplete_seen,
                                usage_seen,
                                attempt.state["upstream_url"],
                            )
                        elif terminal_or_usage_missing:
                            trace_logger.warning(
                                "%s upstream stream finished without completed usage request_id=%s model=%s provider=%s output_seen=%s completed_seen=%s usage_seen=%s upstream_url=%s",
                                self.endpoint,
                                self.request_id,
                                self.request_model_name,
                                attempt.provider_name,
                                output_seen,
                                completed_seen or incomplete_seen,
                                usage_seen,
                                attempt.state["upstream_url"],
                            )
                    finally:
                        if terminal_queue_handoff_completed:
                            cleanup_trigger = "semantic_terminal"
                        elif diagnostics.facts.get("downstream_disconnected"):
                            cleanup_trigger = "downstream_disconnect"
                        elif diagnostics.facts.get("exception_type"):
                            cleanup_trigger = "after_upstream_read_or_stream_failure"
                        else:
                            cleanup_trigger = "consumer_or_shutdown_unknown"
                            diagnostics.mark_local_end(
                                origin="consumer_or_shutdown_unknown"
                            )
                        await _close_observed_responses_upstream_stream_safely(
                            stream_cm,
                            upstream_resp,
                            diagnostics,
                            owner="responses_proxy_finally",
                            trigger=cleanup_trigger,
                        )

    def _finalize_stream_attempt_success(self, attempt: Any) -> None:
        if attempt.state.get("stream_attempt_finalized"):
            return
        attempt.state["stream_attempt_finalized"] = True
        self._record_upstream_attempt_result(
            attempt,
            status_code=int(attempt.state.get("stream_upstream_status_code") or 200),
            success=True,
        )
        finalize_response_memory_attempt(
            attempt=attempt,
            outcome="stream_completed",
        )
        self._mark_success(
            attempt.state["channel_id"],
            attempt.provider_api_key_raw,
        )

    def _finalize_stream_attempt_failure(
        self,
        attempt: Any,
        exc: BaseException,
        *,
        status_code: int = 502,
        error_type: Optional[str] = None,
        terminal_event_type: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        if attempt.state.get("stream_attempt_finalized"):
            return
        attempt.state["stream_attempt_finalized"] = True
        _record_stream_transport_failure(
            self.current_info,
            attempt,
            exc,
        )
        self._record_upstream_attempt_result(
            attempt,
            status_code=status_code,
            success=False,
            error_type=error_type or type(exc).__name__,
            terminal_event_type=terminal_event_type,
            error_code=error_code,
            error_message=exc,
        )
        finalize_response_memory_attempt(
            attempt=attempt,
            outcome="stream_failed",
        )
        if attempt.state.get("track_channel_stats"):
            self._schedule_channel_stats(
                attempt.state["channel_id"],
                success=False,
                provider_api_key=attempt.provider_api_key_raw,
            )

    def _finalize_stream_attempt_cancelled(
        self,
        attempt: Any,
        *,
        downstream_disconnected: bool,
    ) -> None:
        if attempt.state.get("stream_attempt_finalized"):
            return
        attempt.state["stream_attempt_finalized"] = True
        if downstream_disconnected:
            self.current_info["downstream_disconnected"] = True
            self.current_info["stream_outcome"] = "downstream_disconnected"
            self.current_info["error_type"] = "downstream_disconnect"
            self.current_info["success"] = False
        self._record_upstream_attempt_result(
            attempt,
            status_code=499,
            success=False,
            error_type="DownstreamDisconnected",
        )
        finalize_response_memory_attempt(
            attempt=attempt,
            outcome=(
                "downstream_disconnected"
                if downstream_disconnected
                else "cancelled_or_consumer_closed"
            ),
        )

    def _downstream_disconnected(self, attempt: Any, *, stage: str) -> bool:
        if self.disconnect_event is None or not self.disconnect_event.is_set():
            return False
        diagnostics = attempt.state.get("responses_stream_diagnostics_tracker")
        if isinstance(diagnostics, ResponsesStreamDiagnostics):
            diagnostics.mark_downstream_disconnect(stage=stage)
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
        error_text = bounded_stream_error_text(exc)
        request_model, actual_model = _log_model_names(self.request_model_name, attempt.original_model)
        diagnostics = attempt.state.get("responses_stream_diagnostics_tracker")
        diagnostics_json = (
            diagnostics.snapshot_json()
            if isinstance(diagnostics, ResponsesStreamDiagnostics)
            else "{}"
        )
        trace_logger.warning(
            "%s upstream stream aborted stage=%s error_type=%s request_id=%s request_model=%s actual_model=%s provider=%s key=%s upstream_url=%s stream_diagnostics=%s: %s",
            self.endpoint,
            stream_stage,
            type(exc).__name__,
            self.request_id,
            request_model,
            actual_model,
            attempt.provider_name,
            _mask_secret_for_log(attempt.provider_api_key_raw),
            attempt.state["upstream_url"],
            diagnostics_json,
            error_text,
        )

    async def _execute_non_stream_attempt(self, client: Any, attempt: Any, headers: dict[str, str], json_payload: str):
        _mark_current_info_stage(self.current_info, "upstream_send_start")
        runtime_gauges.begin_waiting_first_byte(self.current_info)

        async def cleanup_cancelled_response(response: Any) -> None:
            close = getattr(response, "aclose", None)
            if callable(close):
                await close()

        try:
            upstream_resp = await _await_first_byte_deadline(
                client.post(
                    attempt.state["upstream_url"],
                    headers=headers,
                    content=json_payload,
                    timeout=attempt.state["timeout_value"],
                ),
                disconnect_event=self.disconnect_event,
                cancel_result_cleanup=cleanup_cancelled_response,
            )
            _mark_current_info_stage(self.current_info, "upstream_headers_received")
            _mark_first_byte_observed(self.current_info)
        except DownstreamDisconnectedDuringWait:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            _log_responses_downstream_disconnect(
                self.endpoint,
                self.current_info,
                model_id=self.request_model_name,
                provider_name=attempt.provider_name,
                stage="non-stream-upstream-response",
            )
            return Response(content="", status_code=499)
        except BaseException:
            runtime_gauges.end_waiting_first_byte(self.current_info)
            raise
        if self.disconnect_event is not None and self.disconnect_event.is_set():
            await cleanup_cancelled_response(upstream_resp)
            _log_responses_downstream_disconnect(
                self.endpoint,
                self.current_info,
                model_id=self.request_model_name,
                provider_name=attempt.provider_name,
                stage="non-stream-upstream-response",
            )
            return Response(content="", status_code=499)
        attempt.state["routing_wire_status_code"] = int(
            upstream_resp.status_code
        )
        if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
            raw = await read_limited_response_body(upstream_resp)
            raise HTTPException(status_code=upstream_resp.status_code, detail=raw.text())

        data = await run_json_cpu(json.loads, upstream_resp.content)
        semantic_failure = _responses_failure_http_exception(
            data,
            event_type=str(data.get("type") or "")
            if isinstance(data, dict)
            else None,
            wire_status_code=upstream_resp.status_code,
        )
        if semantic_failure is not None:
            raise semantic_failure

        response_content = upstream_resp.content
        if self._custom_tool_call_id_normalization_enabled(attempt):
            try:
                normalization_result = ResponsesCustomToolCallIdNormalizer().normalize(
                    data
                )
            except ResponsesCustomToolCallIdCollisionError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            if normalization_result.changed:
                response_content = (
                    await run_json_cpu(
                        json.dumps,
                        data,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ).encode("utf-8")
                self._record_custom_tool_call_id_normalization(
                    attempt,
                    normalization_result,
                    direction="output",
                    event_type=str(data.get("type") or "response"),
                )

        self._record_upstream_attempt_result(attempt, status_code=upstream_resp.status_code, success=True)
        self._mark_success(attempt.state["channel_id"], attempt.provider_api_key_raw)
        response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        return Response(
            status_code=upstream_resp.status_code,
            content=response_content,
            headers=response_headers,
            media_type=response_headers.get("content-type", "application/json"),
        )

    def _mark_success(self, channel_id: str, provider_api_key: Optional[str]) -> None:
        self.last_response_failed_terminal = None
        self._schedule_channel_stats(channel_id, success=True, provider_api_key=provider_api_key)
        self.current_info["first_response_time"] = 0
        self.current_info["success"] = True
        self.current_info["provider"] = channel_id

    def _after_failure(self, attempt: Any, exc: Exception, status_code: int, error_message: Any) -> None:
        _record_local_admission_rejection(self.current_info, exc)
        self.last_error_response.clear()
        # A previous retryable provider failure must never leak into the final
        # result of a later attempt.  Retain only one detached, bounded terminal
        # from the current provider-declared or normalized failure terminal.
        self.last_response_failed_terminal = None
        if (
            isinstance(exc, ResponsesSemanticError)
            and isinstance(exc.passthrough_error_body, dict)
        ):
            self.last_error_response.update(exc.passthrough_error_body)
        if (
            isinstance(exc, ResponsesSemanticError)
            and exc.responses_sse_event_type == "response.failed"
        ):
            self.last_response_failed_terminal = CachedResponsesFailureTerminal(
                data=_encode_responses_sse_event(
                    exc.responses_sse_event_type,
                    exc.responses_sse_payload,
                ),
                status_code=int(exc.status_code),
                error_code=exc.error_code,
                error_type=exc.error_type,
            )
        self._record_upstream_attempt_result(
            attempt,
            status_code=status_code,
            success=False,
            error_type=type(exc).__name__,
            terminal_event_type=getattr(exc, "event_type", None),
            error_code=getattr(exc, "error_code", None),
            error_message=error_message,
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
        if self.last_error_response:
            return JSONResponse(
                status_code=status_code,
                content=self.last_error_response,
            )
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
        request_body_bytes = _request_body_size_bytes(http_request, request_body)
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=endpoint,
            request_body_bytes=request_body_bytes,
            reasoning_effort=request_reasoning_effort(request_body),
            debug=is_debug,
            provider_resolver=get_right_order_providers,
        )
        _record_plan_observability(current_info, plan)
        runner = UpstreamRunner(
            plan,
            endpoint=endpoint,
            debug=is_debug,
            observability_context=current_info,
        )
        replay_store = MessagesRequestReplayStore(request_body_bytes)
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
            "request_replay_store": replay_store,
        }

        try:
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
        finally:
            await replay_store.aclose()

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
        apply_oaix_routing_attempt_id(
            headers,
            provider=attempt.provider,
            routing_attempt_id=attempt.routing_attempt_id,
        )
        self._messages_log_attempt(ctx, attempt, payload, headers)
        replay_store = ctx["request_replay_store"]
        prepared_body = await replay_store.prepare(
            (
                attempt.provider_name,
                original_model,
                bool(payload.get("stream")),
                upstream_url,
                id(provider),
            ),
            payload,
        )
        headers["Content-Length"] = str(prepared_body.content_length)
        attempt.state["request_body_storage"] = prepared_body.storage
        attempt.state["request_body_wire_bytes"] = prepared_body.content_length

        try:
            async with app.state.client_manager.get_client(upstream_url, proxy) as client:
                if payload.get("stream"):
                    return await self._messages_stream_response(
                        client,
                        attempt,
                        ctx,
                        headers,
                        prepared_body.content,
                    )
                return await self._messages_non_stream_response(
                    client,
                    attempt,
                    ctx,
                    headers,
                    prepared_body.content,
                )
        finally:
            await prepared_body.aclose()

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

    async def _messages_stream_response(self, client: Any, attempt: Any, ctx: dict[str, Any], headers: dict[str, str], json_payload: Any):
        upstream_url = attempt.state["upstream_url"]
        stream_cm = client.stream("POST", upstream_url, headers=headers, content=json_payload, timeout=attempt.state["timeout_value"])

        async def cleanup_cancelled_stream_enter(response: Any) -> None:
            await _close_upstream_response_stream_safely(stream_cm, response)

        try:
            upstream_resp = await _await_first_byte_deadline(
                stream_cm.__aenter__(),
                disconnect_event=ctx["disconnect_event"],
                cancel_result_cleanup=cleanup_cancelled_stream_enter,
            )
        except DownstreamDisconnectedDuringWait:
            trace_logger.info(
                "%s downstream disconnect stage=upstream-response-headers request_id=%s model=%s provider=%s",
                ctx["endpoint"],
                ctx["request_id"],
                ctx["request_model_name"],
                attempt.provider_name,
            )
            return Response(content="", status_code=499)
        response_headers = _copy_upstream_response_headers(upstream_resp.headers)
        if upstream_resp.status_code < 200 or upstream_resp.status_code >= 300:
            raw = await read_limited_response_body(upstream_resp)
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            self._messages_set_last_error(ctx, raw.body, response_headers)
            raise HTTPException(status_code=upstream_resp.status_code, detail=raw.text())

        upstream_iter = upstream_resp.aiter_raw()
        try:
            buffered_chunks = await _prime_passthrough_upstream_stream(upstream_iter, disconnect_event=ctx["disconnect_event"])
        except DownstreamDisconnectedDuringWait:
            await _close_upstream_response_stream_safely(stream_cm, upstream_resp)
            trace_logger.info(
                "%s downstream disconnect stage=before-stream-commit request_id=%s model=%s provider=%s",
                ctx["endpoint"],
                ctx["request_id"],
                ctx["request_model_name"],
                attempt.provider_name,
            )
            return Response(content="", status_code=499)
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

        attempt.state["stream_upstream_status_code"] = upstream_resp.status_code
        return StarletteStreamingResponse(
            self._messages_proxy_stream(ctx, attempt, buffered_chunks, upstream_iter, stream_cm, upstream_resp),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            media_type=response_headers.get("content-type", "text/event-stream"),
        )

    async def _messages_proxy_stream(self, ctx: dict[str, Any], attempt: Any, buffered_chunks: list[bytes], upstream_iter: Any, stream_cm: Any, upstream_resp: Any):
        parser = IncrementalSSEParser()
        terminal_seen = False

        async def source_events():
            try:
                while buffered_chunks:
                    chunk = buffered_chunks.pop(0)
                    raw_events = parser.feed(bytes(chunk))
                    chunk = None
                    try:
                        for event_index in range(len(raw_events)):
                            raw_event = raw_events[event_index]
                            try:
                                yield raw_event
                            finally:
                                raw_event = None
                                raw_events[event_index] = ""
                    finally:
                        raw_events.clear()
                        raw_events = None
                buffered_chunks.clear()
            finally:
                # Ensure a terminal return cannot leave the precommit list's
                # remaining byte aliases to async-generator GC finalization.
                buffered_chunks.clear()
            while True:
                try:
                    chunk = await _await_first_byte_deadline(
                        upstream_iter.__anext__(),
                        disconnect_event=ctx["disconnect_event"],
                    )
                except StopAsyncIteration:
                    break
                raw_events = parser.feed(bytes(chunk))
                chunk = None
                try:
                    for event_index in range(len(raw_events)):
                        raw_event = raw_events[event_index]
                        try:
                            yield raw_event
                        finally:
                            raw_event = None
                            raw_events[event_index] = ""
                finally:
                    raw_events.clear()
                    raw_events = None
            raw_events = parser.finish()
            try:
                for event_index in range(len(raw_events)):
                    raw_event = raw_events[event_index]
                    try:
                        yield raw_event
                    finally:
                        raw_event = None
                        raw_events[event_index] = ""
            finally:
                raw_events.clear()
                raw_events = None

        source = source_events()
        try:
            async for raw_event in source:
                if self._messages_downstream_disconnected(ctx, attempt, stage="after-stream-commit"):
                    self._messages_finalize_stream_disconnect(ctx, attempt)
                    return
                if not raw_event.strip():
                    raw_event = None
                    continue
                event_bytes = _raw_responses_sse_event_bytes(raw_event)
                if is_sse_comment_frame(raw_event):
                    try:
                        yield event_bytes
                    finally:
                        event_bytes = None
                        raw_event = None
                    continue
                event_owner = await parse_owned_sse_event(raw_event)
                event_name = event_owner.event_name
                payload = event_owner.payload
                payload_type = None
                try:
                    if not isinstance(payload, dict):
                        raise SSEProtocolError(
                            "Anthropic upstream SSE data must be a JSON object"
                        )
                    payload_type = str(payload.get("type") or "").strip()
                    if event_name == "message_stop" or payload_type == "message_stop":
                        terminal_seen = True
                    yield event_bytes
                    if terminal_seen:
                        # message_stop is the protocol terminal.  Waiting for
                        # EOF on a keep-alive connection would retain leases.
                        self._messages_finalize_stream_success(ctx, attempt)
                        return
                finally:
                    payload_type = None
                    payload = None
                    event_name = None
                    raw_event = None
                    event_bytes = None
                    owner_to_close = event_owner
                    event_owner = None
                    await owner_to_close.aclose()
                    owner_to_close = None
            raise SSEProtocolError(
                "Anthropic upstream ended without message_stop"
            )
        except DownstreamDisconnectedDuringWait:
            self._messages_finalize_stream_disconnect(ctx, attempt)
            return
        except SSEProtocolError as exc:
            self._messages_finalize_stream_failure(ctx, attempt, exc)
            _record_postcommit_sse_protocol_error_isolation(
                ctx["current_info"],
                exc,
            )
            yield _postcommit_sse_protocol_error_event()
            return
        except UPSTREAM_NETWORK_ERRORS as exc:
            self._messages_finalize_stream_failure(ctx, attempt, exc)
            raise
        except AdmissionRejected as exc:
            _record_local_admission_rejection(ctx["current_info"], exc)
            ctx["current_info"]["stream_outcome"] = "local_backpressure_abort"
            ctx["current_info"]["stream_error_status_code"] = int(
                getattr(exc, "status_code", 503)
            )
            attempt.state["messages_stream_finalized"] = True
            raise
        except (asyncio.CancelledError, GeneratorExit):
            if ctx["disconnect_event"] is not None and ctx["disconnect_event"].is_set():
                self._messages_finalize_stream_disconnect(ctx, attempt)
            raise
        finally:
            try:
                await source.aclose()
            finally:
                parser.discard()
                buffered_chunks.clear()
                try:
                    if (
                        not attempt.state.get("messages_stream_finalized")
                        and ctx["disconnect_event"] is not None
                        and ctx["disconnect_event"].is_set()
                    ):
                        self._messages_finalize_stream_disconnect(ctx, attempt)
                finally:
                    await _close_upstream_response_stream_safely(
                        stream_cm,
                        upstream_resp,
                    )

    def _messages_finalize_stream_success(
        self,
        ctx: dict[str, Any],
        attempt: Any,
    ) -> None:
        if attempt.state.get("messages_stream_finalized"):
            return
        attempt.state["messages_stream_finalized"] = True
        self._messages_record_success(ctx, attempt)

    def _messages_finalize_stream_failure(
        self,
        ctx: dict[str, Any],
        attempt: Any,
        exc: BaseException,
    ) -> None:
        if attempt.state.get("messages_stream_finalized"):
            return
        attempt.state["messages_stream_finalized"] = True
        current_info = ctx["current_info"]
        _record_stream_transport_failure(current_info, attempt, exc)
        current_info["success"] = False
        current_info["stream_outcome"] = "upstream_stream_abort"
        current_info["stream_error_status_code"] = 502
        current_info["error_type"] = type(exc).__name__
        if attempt.state.get("track_channel_stats"):
            _schedule_channel_stats_bounded(
                current_info["request_id"],
                attempt.state["channel_id"],
                ctx["request_model_name"],
                current_info["api_key"],
                success=False,
                provider_api_key=attempt.provider_api_key_raw,
                fallback_background_tasks=ctx["background_tasks"],
            )

    def _messages_finalize_stream_disconnect(
        self,
        ctx: dict[str, Any],
        attempt: Any,
    ) -> None:
        if attempt.state.get("messages_stream_finalized"):
            return
        attempt.state["messages_stream_finalized"] = True
        current_info = ctx["current_info"]
        current_info["downstream_disconnected"] = True
        current_info["stream_outcome"] = "downstream_disconnected"
        current_info["error_type"] = "downstream_disconnect"
        current_info["success"] = False

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

    async def _messages_non_stream_response(self, client: Any, attempt: Any, ctx: dict[str, Any], headers: dict[str, str], json_payload: Any):
        async def cleanup_cancelled_response(response: Any) -> None:
            close = getattr(response, "aclose", None)
            if callable(close):
                await close()

        try:
            upstream_resp = await _await_first_byte_deadline(
                client.post(
                    attempt.state["upstream_url"],
                    headers=headers,
                    content=json_payload,
                    timeout=attempt.state["timeout_value"],
                ),
                disconnect_event=ctx["disconnect_event"],
                cancel_result_cleanup=cleanup_cancelled_response,
            )
        except DownstreamDisconnectedDuringWait:
            trace_logger.info(
                "%s downstream disconnect stage=non-stream-upstream-response request_id=%s model=%s provider=%s",
                ctx["endpoint"],
                ctx["request_id"],
                ctx["request_model_name"],
                attempt.provider_name,
            )
            return Response(content="", status_code=499)
        if ctx["disconnect_event"] is not None and ctx["disconnect_event"].is_set():
            await cleanup_cancelled_response(upstream_resp)
            trace_logger.info(
                "%s downstream disconnect stage=non-stream-upstream-response request_id=%s model=%s provider=%s",
                ctx["endpoint"],
                ctx["request_id"],
                ctx["request_model_name"],
                attempt.provider_name,
            )
            return Response(content="", status_code=499)
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
        _schedule_channel_stats_bounded(
            current_info["request_id"],
            channel_id,
            ctx["request_model_name"],
            current_info["api_key"],
            success=True,
            provider_api_key=attempt.provider_api_key_raw,
            fallback_background_tasks=ctx["background_tasks"],
        )
        current_info["first_response_time"] = 0
        current_info["success"] = True
        current_info["provider"] = channel_id

    def _messages_after_failure(self, attempt: Any, exc: Exception, status_code: int, error_message: Any, ctx: dict[str, Any]) -> None:
        current_info = ctx["current_info"]
        _record_local_admission_rejection(current_info, exc)
        if attempt.state.get("track_channel_stats"):
            _schedule_channel_stats_bounded(
                current_info["request_id"],
                attempt.state["channel_id"],
                ctx["request_model_name"],
                current_info["api_key"],
                success=False,
                provider_api_key=attempt.provider_api_key_raw,
                fallback_background_tasks=ctx["background_tasks"],
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
                json_payload = await run_json_cpu(json.dumps, payload or {})
                return await client.post(upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
            if method == "GET":
                return await client.get(upstream_url, headers=headers, timeout=timeout_value)
            if method == "PUT":
                json_payload = await run_json_cpu(json.dumps, payload or {})
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
        _schedule_channel_stats_bounded(
            current_info["request_id"],
            channel_id,
            request_model_name,
            current_info["api_key"],
            success=success,
            provider_api_key=provider_api_key_raw,
            fallback_background_tasks=background_tasks,
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
            disconnect_event = current_info.get("disconnect_event")
            upstream_resp = await _await_buffered_upstream_or_disconnect(
                self._send_upstream(
                    method=upstream_request.method,
                    upstream_url=upstream_url,
                    headers=headers,
                    payload=upstream_request.payload,
                    proxy=proxy,
                    timeout_value=int(timeout_resolution["timeout_value"]),
                ),
                disconnect_event,
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
        except DownstreamDisconnectedDuringWait:
            return Response(content="", status_code=499)
        except Exception as exc:
            if not _record_local_admission_rejection(current_info, exc):
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
        config = app.state.config
        if not api_key_has_model_rules(app, api_index):
            raise HTTPException(status_code=404, detail=f"No matching model found: {request_model_name}")

        current_info = get_request_info()
        current_info["model"] = current_info.get("model") or request_model_name
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_id = _responses_request_id(current_info)
        request_body_bytes = _request_body_size_bytes(http_request, request_body)
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
            request_body_bytes=request_body_bytes,
            reasoning_effort=request_reasoning_effort(request_body),
            debug=is_debug,
            provider_resolver=self._provider_resolver(request_body),
        )
        _record_plan_observability(current_info, plan)
        runner = UpstreamRunner(
            plan,
            endpoint=CONTENT_GENERATION_TASKS_ENDPOINT,
            debug=is_debug,
            observability_context=current_info,
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
        try:
            upstream_resp = await _await_buffered_upstream_or_disconnect(
                self._send_upstream(
                    method=upstream_request.method,
                    upstream_url=attempt.state["upstream_url"],
                    headers=upstream_request.headers,
                    payload=payload,
                    proxy=attempt.state["proxy"],
                    timeout_value=attempt.state["timeout_value"],
                ),
                ctx["disconnect_event"],
            )
        except DownstreamDisconnectedDuringWait:
            return Response(content="", status_code=499)
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
        _record_local_admission_rejection(current_info, exc)
        if attempt.state.get("track_channel_stats"):
            _schedule_channel_stats_bounded(
                current_info["request_id"],
                attempt.state["channel_id"],
                ctx["request_model_name"],
                current_info["api_key"],
                success=False,
                provider_api_key=attempt.provider_api_key_raw,
                fallback_background_tasks=ctx["background_tasks"],
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
                json_payload = await run_json_cpu(json.dumps, payload or {})
                return await client.post(upstream_url, headers=headers, content=json_payload, timeout=timeout_value)
            if method == "PUT":
                json_payload = await run_json_cpu(json.dumps, payload or {})
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
        disconnect_event = current_info.get("disconnect_event") if isinstance(current_info, dict) else None
        request_id = _responses_request_id(current_info)
        request_body_bytes = _request_body_size_bytes(http_request, payload)

        plan = await RoutingPlan.create(
            app,
            request_model_name,
            api_index,
            self.last_provider_indices,
            self.locks,
            endpoint=endpoint,
            request_body_bytes=request_body_bytes,
            reasoning_effort=request_reasoning_effort(payload),
            debug=is_debug,
            provider_resolver=get_right_order_providers,
        )
        _record_plan_observability(current_info, plan)
        runner = UpstreamRunner(
            plan,
            endpoint=endpoint,
            debug=is_debug,
            observability_context=current_info,
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
            "disconnect_event": disconnect_event,
            "request_id": request_id,
            "plan": plan,
            "runner": runner,
            "background_tasks": background_tasks,
            "last_error_response": {},
        }

        return await runner.run(
            lambda attempt: self._lingjing_execute_attempt(attempt, ctx),
            prepare_attempt=lambda attempt: self._lingjing_prepare_attempt(attempt, ctx),
            before_next_attempt=lambda: self._lingjing_before_next_attempt(ctx),
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

    async def _lingjing_before_next_attempt(self, ctx: dict[str, Any]):
        disconnect_event = ctx["disconnect_event"]
        if disconnect_event is not None and disconnect_event.is_set():
            return Response(content="", status_code=499)
        return None

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
        try:
            upstream_resp = await _await_buffered_upstream_or_disconnect(
                self._send_upstream(
                    method=ctx["method_upper"],
                    upstream_url=attempt.state["upstream_url"],
                    headers=headers,
                    payload=outbound_payload,
                    proxy=attempt.state["proxy"],
                    timeout_value=attempt.state["timeout_value"],
                ),
                ctx["disconnect_event"],
            )
        except DownstreamDisconnectedDuringWait:
            return Response(content="", status_code=499)
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
        _schedule_channel_stats_bounded(
            current_info["request_id"],
            channel_id,
            ctx["request_model_name"],
            current_info["api_key"],
            success=success,
            provider_api_key=attempt.provider_api_key_raw,
            fallback_background_tasks=ctx["background_tasks"],
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
        _record_local_admission_rejection(current_info, exc)
        if attempt.state.get("track_channel_stats"):
            _schedule_channel_stats_bounded(
                current_info["request_id"],
                attempt.state["channel_id"],
                ctx["request_model_name"],
                current_info["api_key"],
                success=False,
                provider_api_key=attempt.provider_api_key_raw,
                fallback_background_tasks=ctx["background_tasks"],
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
alpha_search_handler = AlphaSearchRequestHandler(
    app=app,
    get_runtime_api_list=get_runtime_api_list,
    api_key_has_model_rules=api_key_has_model_rules,
    resolve_codex_upstream_auth=_resolve_codex_upstream_auth,
    resolve_timeout=_resolve_alpha_search_timeout,
    add_trace_headers=_add_trace_headers,
    record_plan_observability=_record_plan_observability,
    record_retry_observability=_record_retry_observability,
    provider_resolver=get_right_order_providers,
    debug=lambda: is_debug,
)
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
    await _drop_parsed_request_caches(http_request)
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
    await _drop_parsed_request_caches(http_request)
    return await responses_api_response(
        responses_handler=responses_handler,
        http_request=http_request,
        request=request,
        background_tasks=background_tasks,
        api_index=api_index,
    )


@app.post(ALPHA_SEARCH_ENDPOINT, dependencies=[Depends(rate_limit_dependency)])
async def alpha_search_route(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request_body: Any = Body(...),
    api_index: int = Depends(verify_api_key),
):
    return await alpha_search_handler.request_search(
        http_request=http_request,
        request_body=request_body,
        api_index=api_index,
        background_tasks=background_tasks,
    )

@app.post("/v1/responses/compact", dependencies=[Depends(rate_limit_dependency)])
async def responses_compact_route(
    http_request: Request,
    request: ResponsesRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    await _drop_parsed_request_caches(http_request)
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
    await _drop_parsed_request_caches(http_request)
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
async def list_models(
    client_version: Optional[str] = None,
    api_index: int = Depends(verify_api_key),
):
    runtime_api_list = get_runtime_api_list()
    if str(client_version or "").strip():
        return JSONResponse(
            content=codex_models_payload(
                api_index=api_index,
                api_list=runtime_api_list,
                model_response_cache=getattr(app.state, "model_response_cache", {}) or {},
                config=app.state.config,
                models_list=app.state.models_list,
                build_models=post_all_models,
            ),
            headers={
                "X-Uni-API-Models-Source": "codex-pro-snapshot",
                "X-Uni-API-Models-Snapshot-Client-Version": CODEX_PRO_MODELS_SNAPSHOT_CLIENT_VERSION,
                "X-Uni-API-Models-Upstream-ETag": CODEX_PRO_MODELS_SNAPSHOT_UPSTREAM_ETAG,
            },
        )
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
    http_request: Request,
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await image_generation_response(model_handler, request, api_index, background_tasks, http_request=http_request)

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
    http_request: Request,
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await embeddings_response(model_handler, request, api_index, background_tasks, http_request=http_request)

@app.post("/v1/audio/speech", dependencies=[Depends(rate_limit_dependency)])
async def audio_speech(
    http_request: Request,
    request: TextToSpeechRequest,
    background_tasks: BackgroundTasks,
    api_index: str = Depends(verify_api_key)
):
    return await audio_speech_response(model_handler, request, api_index, background_tasks, http_request=http_request)

@app.post("/v1/moderations", dependencies=[Depends(rate_limit_dependency)])
async def moderations(
    http_request: Request,
    request: ModerationRequest,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key)
):
    return await moderation_response(model_handler, request, api_index, background_tasks, http_request=http_request)

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
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "static"), html=True), name="static")

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
        http=UVICORN_HTTP_PROTOCOL,
        limit_concurrency=None,
        backlog=UVICORN_BACKLOG,
        # log_level="warning"
    )
