from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from time import time
from typing import Any

import httpx


logger = logging.getLogger("uni-api")

_TRACE_ENDPOINT = "/v1/traces"
_LOG_ENDPOINT = "/v1/logs"
_METRIC_ENDPOINT = "/v1/metrics"
_DEFAULT_SERVICE_NAME = "uni-api-ember"
_DEFAULT_QUEUE_MAX_SIZE = 10000
_DEFAULT_EXPORT_WORKER_COUNT = 4
_DEFAULT_EXPORT_TIMEOUT_SECONDS = 2.0
_DEFAULT_SAMPLE_RATE = 1.0
_REQUEST_BODY_COMPLEXITY_REASONS = frozenset(
    {"max_depth", "max_scalar_bytes", "max_estimated_bytes"}
)
_REQUEST_BODY_COMPLEXITY_TRIGGER_PHASES = frozenset(
    {
        "chunk_raw_charge",
        "structural_item_scan",
        "depth_scan",
        "scalar_scan",
    }
)
_WORKER_PERFORMANCE_PHASES = (
    "socket_receive",
    "sse_frame",
    "json_parse",
    "observer_hash",
    "queue_put",
    "asgi_write",
    "idempotency_hash",
)
_WORKER_PHASE_SAMPLE_SCOPES = (
    "default",
    "responses_stream",
    "idempotency_hash",
)
_WORKER_THREADPOOL_TASK_CATEGORIES = (
    "json_parse",
    "json_serialization",
    "network_procfs",
    "on_cpu_profile",
    "request_body_decode",
    "upstream_response_decode",
    "other",
)
_WORKER_THREADPOOL_CATEGORY_SOURCES = frozenset(
    {
        "explicit_task_tag",
        "dedicated_thread_name",
        "default_executor_stack",
    }
)
_WORKER_THREADPOOL_LIFECYCLE_SEMANTICS = (
    "explicit_task_tag_wall_thread_cpu_v1"
)
_WORKER_THREADPOOL_PROFILE_SEMANTICS = (
    "explicit_tag_then_dedicated_name_then_bounded_stack_v1"
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
_TERMINAL_TIMELINE_TRANSITIONS = (
    ("received", "parse_completed"),
    ("parse_completed", "observer_completed"),
    ("observer_completed", "semantic_classified"),
    ("semantic_classified", "queue_handoff_completed"),
    ("queue_handoff_completed", "asgi_write_attempted"),
    ("asgi_write_attempted", "asgi_write_completed"),
)

_STAGE_ORDER = [
    "request_received",
    "body_parsed",
    "provider_selected",
    "provider_key_selected",
    "retry_started",
    "client_pool_acquired",
    "upstream_send_start",
    "upstream_headers_received",
    "upstream_first_chunk",
    "downstream_response_start",
    "stream_end",
]

_TRANSPORT_PHASE_FIELDS = (
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
)


@dataclass(frozen=True)
class FugueObservabilityConfig:
    endpoint: str | None
    service_name: str = _DEFAULT_SERVICE_NAME
    service_version: str | None = None
    queue_max_size: int = _DEFAULT_QUEUE_MAX_SIZE
    export_worker_count: int = _DEFAULT_EXPORT_WORKER_COUNT
    export_timeout_seconds: float = _DEFAULT_EXPORT_TIMEOUT_SECONDS
    sample_rate: float = _DEFAULT_SAMPLE_RATE
    identity_attrs: dict[str, str] = field(default_factory=dict)
    emit_request_summaries: bool = True
    emit_stage_spans: bool = True
    emit_metrics: bool = True

    @property
    def enabled(self) -> bool:
        return bool((self.endpoint or "").strip())


@dataclass(frozen=True)
class _DeferredLargeBodyAdmissionDecision:
    decision: Any


@dataclass(frozen=True)
class _DeferredAdmission503ResponseWriteOutcome:
    outcome: Any


@dataclass(frozen=True)
class _DeferredResponseBufferEvent:
    event: Any


@dataclass(frozen=True)
class _DeferredWorkerRuntimeSnapshot:
    snapshot: dict[str, Any]


@dataclass(frozen=True)
class _DeferredWorkerCPUProfile:
    profile: dict[str, Any]


@dataclass(frozen=True)
class _DeferredTerminalHopObservation:
    observation: dict[str, Any]


class FugueObservabilityClient:
    def __init__(self, config: FugueObservabilityConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[
            tuple[str, dict[str, Any]]
            | _DeferredLargeBodyAdmissionDecision
            | _DeferredAdmission503ResponseWriteOutcome
            | _DeferredResponseBufferEvent
            | _DeferredWorkerRuntimeSnapshot
            | _DeferredWorkerCPUProfile
            | _DeferredTerminalHopObservation
        ] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._client: httpx.AsyncClient | None = None
        self._dropped = 0
        self._export_errors = 0
        self._large_body_decision_enqueued = 0
        self._large_body_decision_enqueue_dropped = 0
        self._large_body_decision_build_errors = 0
        self._large_body_decision_export_errors = 0
        self._admission_503_outcome_enqueued = 0
        self._admission_503_outcome_enqueue_dropped = 0
        self._admission_503_outcome_build_errors = 0
        self._admission_503_outcome_export_errors = 0
        self._response_buffer_event_enqueued = 0
        self._response_buffer_event_enqueue_dropped = 0
        self._response_buffer_event_build_errors = 0
        self._response_buffer_event_export_errors = 0
        self._worker_runtime_snapshot_enqueued = 0
        self._worker_runtime_snapshot_enqueue_dropped = 0
        self._worker_runtime_snapshot_build_errors = 0
        self._worker_runtime_snapshot_export_errors = 0
        self._worker_cpu_profile_enqueued = 0
        self._worker_cpu_profile_enqueue_dropped = 0
        self._worker_cpu_profile_build_errors = 0
        self._worker_cpu_profile_export_errors = 0
        self._terminal_hop_observation_enqueued = 0
        self._terminal_hop_observation_enqueue_dropped = 0
        self._terminal_hop_observation_build_errors = 0
        self._terminal_hop_observation_export_errors = 0

    async def start(self) -> None:
        if not self.config.enabled or self._tasks:
            return
        self._queue = asyncio.Queue(maxsize=max(1, int(self.config.queue_max_size)))
        self._client = httpx.AsyncClient(timeout=self.config.export_timeout_seconds)
        worker_count = max(1, int(self.config.export_worker_count))
        self._tasks = [
            asyncio.create_task(
                self._worker(),
                name=f"uni-api-ember-fugue-observability-exporter-{index}",
            )
            for index in range(worker_count)
        ]
        logger.info(
            "Fugue observability exporter enabled for service=%s workers=%s queue_max_size=%s",
            self.config.service_name,
            worker_count,
            self.config.queue_max_size,
        )

    async def stop(self) -> None:
        tasks = self._tasks
        self._tasks = []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()
        self._queue = None

    def emit_request(self, *, current_info: dict[str, Any], runtime_metrics: dict[str, Any] | None = None) -> None:
        if not self.config.enabled:
            return
        status_code = _safe_int(current_info.get("status_code"), 0)
        stream_failure = _is_stream_failure(current_info)
        downstream_disconnected = _safe_bool(
            current_info.get("downstream_disconnected")
        ) or status_code == 499
        retain_for_responses_correlation = _is_responses_request(current_info)
        sampled_out = (
            status_code < 400
            and not stream_failure
            and not downstream_disconnected
            and self.config.sample_rate < 1.0
            and (
                self.config.sample_rate <= 0.0
                or random.random() > self.config.sample_rate
            )
        )
        if sampled_out and not retain_for_responses_correlation:
            return
        telemetry = build_uni_api_ember_request_telemetry(
            service_name=self.config.service_name,
            service_version=self.config.service_version,
            identity_attrs=self.config.identity_attrs,
            current_info=current_info,
            runtime_metrics=runtime_metrics,
        )
        if self.config.emit_request_summaries:
            self._emit_events(_LOG_ENDPOINT, telemetry["logs"])
        # When ordinary success sampling excludes a Responses request, retain
        # only its compact correlation logs.  Stage spans and per-request
        # metric batches must not crowd those logs out of the bounded queue.
        if self.config.emit_stage_spans and not sampled_out:
            self._emit_events(_TRACE_ENDPOINT, telemetry["traces"])
        if self.config.emit_metrics and not sampled_out:
            self._emit_events(_METRIC_ENDPOINT, telemetry["metrics"])

    def emit_large_body_admission_decision(self, decision: Any) -> bool | None:
        if not self.config.enabled:
            return None
        accepted = self._enqueue(
            _DeferredLargeBodyAdmissionDecision(decision),
            event_count=1,
        )
        if accepted:
            self._large_body_decision_enqueued += 1
        else:
            self._large_body_decision_enqueue_dropped += 1
        return accepted

    def emit_admission_503_response_write_outcome(
        self,
        outcome: Any,
    ) -> bool | None:
        if not self.config.enabled:
            return None
        accepted = self._enqueue(
            _DeferredAdmission503ResponseWriteOutcome(outcome),
            event_count=1,
        )
        if accepted:
            self._admission_503_outcome_enqueued += 1
        else:
            self._admission_503_outcome_enqueue_dropped += 1
        return accepted

    def emit_response_buffer_event(self, event: Any) -> bool | None:
        if not self.config.enabled:
            return None
        accepted = self._enqueue(
            _DeferredResponseBufferEvent(event),
            event_count=1,
        )
        if accepted:
            self._response_buffer_event_enqueued += 1
        else:
            self._response_buffer_event_enqueue_dropped += 1
        return accepted

    def emit_worker_runtime_snapshot(
        self,
        snapshot: dict[str, Any],
    ) -> bool | None:
        if not self.config.enabled:
            return None
        accepted = self._enqueue(
            _DeferredWorkerRuntimeSnapshot(dict(snapshot)),
            event_count=1,
        )
        if accepted:
            self._worker_runtime_snapshot_enqueued += 1
        else:
            self._worker_runtime_snapshot_enqueue_dropped += 1
        return accepted

    def emit_worker_cpu_profile(
        self,
        profile: dict[str, Any],
    ) -> bool | None:
        if not self.config.enabled:
            return None
        accepted = self._enqueue(
            _DeferredWorkerCPUProfile(dict(profile)),
            event_count=1,
        )
        if accepted:
            self._worker_cpu_profile_enqueued += 1
        else:
            self._worker_cpu_profile_enqueue_dropped += 1
        return accepted

    def emit_terminal_hop_observation(
        self,
        observation: dict[str, Any],
    ) -> bool | None:
        if not self.config.enabled:
            return None
        accepted = self._enqueue(
            _DeferredTerminalHopObservation(dict(observation)),
            event_count=1,
        )
        if accepted:
            self._terminal_hop_observation_enqueued += 1
        else:
            self._terminal_hop_observation_enqueue_dropped += 1
        return accepted

    def _emit_events(
        self,
        path: str,
        events: list[dict[str, Any]],
    ) -> bool | None:
        if not events:
            return None
        return self._enqueue((path, {"events": events}), event_count=len(events))

    def _enqueue(self, item: Any, *, event_count: int) -> bool:
        queue = self._queue
        if queue is None:
            return False
        try:
            queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self._dropped += event_count
            if self._dropped == event_count or self._dropped % 100 == 0:
                logger.warning("Fugue observability queue full; dropped %s event(s)", self._dropped)
            return False

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            large_body_decision = isinstance(
                item,
                _DeferredLargeBodyAdmissionDecision,
            )
            write_outcome = isinstance(
                item,
                _DeferredAdmission503ResponseWriteOutcome,
            )
            response_buffer_event = isinstance(
                item,
                _DeferredResponseBufferEvent,
            )
            worker_runtime_snapshot = isinstance(
                item,
                _DeferredWorkerRuntimeSnapshot,
            )
            worker_cpu_profile = isinstance(
                item,
                _DeferredWorkerCPUProfile,
            )
            terminal_hop_observation = isinstance(
                item,
                _DeferredTerminalHopObservation,
            )
            build_failed = False
            posts: list[tuple[str, dict[str, Any]]] | None = None
            try:
                if large_body_decision:
                    try:
                        event = build_uni_api_ember_large_body_admission_event(
                            service_name=self.config.service_name,
                            service_version=self.config.service_version,
                            identity_attrs=self.config.identity_attrs,
                            decision=item.decision,
                        )
                    except Exception:
                        build_failed = True
                        self._export_errors += 1
                        self._large_body_decision_build_errors += 1
                        raise
                    path, payload = _LOG_ENDPOINT, {"events": [event]}
                elif write_outcome:
                    try:
                        event = build_uni_api_ember_admission_503_response_write_event(
                            service_name=self.config.service_name,
                            service_version=self.config.service_version,
                            identity_attrs=self.config.identity_attrs,
                            outcome=item.outcome,
                        )
                    except Exception:
                        build_failed = True
                        self._export_errors += 1
                        self._admission_503_outcome_build_errors += 1
                        raise
                    path, payload = _LOG_ENDPOINT, {"events": [event]}
                elif response_buffer_event:
                    try:
                        event = build_uni_api_ember_response_buffer_event(
                            service_name=self.config.service_name,
                            service_version=self.config.service_version,
                            identity_attrs=self.config.identity_attrs,
                            response_event=item.event,
                        )
                    except Exception:
                        build_failed = True
                        self._export_errors += 1
                        self._response_buffer_event_build_errors += 1
                        raise
                    path, payload = _LOG_ENDPOINT, {"events": [event]}
                elif worker_runtime_snapshot:
                    try:
                        events = build_uni_api_ember_worker_metric_events(
                            service_name=self.config.service_name,
                            service_version=self.config.service_version,
                            identity_attrs=self.config.identity_attrs,
                            snapshot=item.snapshot,
                        )
                        snapshot_event = (
                            build_uni_api_ember_worker_runtime_snapshot_event(
                                service_name=self.config.service_name,
                                service_version=self.config.service_version,
                                identity_attrs=self.config.identity_attrs,
                                snapshot=item.snapshot,
                            )
                        )
                    except Exception:
                        build_failed = True
                        self._export_errors += 1
                        self._worker_runtime_snapshot_build_errors += 1
                        raise
                    # Persist the per-worker snapshot in app_events first.
                    # Some Fugue installations intentionally run without a
                    # Prometheus remote-write backend; the structured event
                    # keeps the same facts queryable in that configuration.
                    posts = [
                        (_LOG_ENDPOINT, {"events": [snapshot_event]}),
                        (_METRIC_ENDPOINT, {"events": events}),
                    ]
                elif worker_cpu_profile:
                    try:
                        event = build_uni_api_ember_worker_cpu_profile_event(
                            service_name=self.config.service_name,
                            service_version=self.config.service_version,
                            identity_attrs=self.config.identity_attrs,
                            profile=item.profile,
                        )
                    except Exception:
                        build_failed = True
                        self._export_errors += 1
                        self._worker_cpu_profile_build_errors += 1
                        raise
                    path, payload = _LOG_ENDPOINT, {"events": [event]}
                elif terminal_hop_observation:
                    try:
                        event = build_uni_api_ember_terminal_hop_metric_event(
                            service_name=self.config.service_name,
                            service_version=self.config.service_version,
                            identity_attrs=self.config.identity_attrs,
                            observation=item.observation,
                        )
                    except Exception:
                        build_failed = True
                        self._export_errors += 1
                        self._terminal_hop_observation_build_errors += 1
                        raise
                    path, payload = _METRIC_ENDPOINT, {"events": [event]}
                else:
                    path, payload = item
                if posts is None:
                    posts = [(path, payload)]
                post_errors: list[Exception] = []
                for post_path, post_payload in posts:
                    if not post_payload.get("events"):
                        continue
                    try:
                        await self._post_json(post_path, post_payload)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        post_errors.append(exc)
                if post_errors:
                    raise post_errors[0]
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not build_failed:
                    self._export_errors += 1
                if large_body_decision and not build_failed:
                    self._large_body_decision_export_errors += 1
                if write_outcome and not build_failed:
                    self._admission_503_outcome_export_errors += 1
                if response_buffer_event and not build_failed:
                    self._response_buffer_event_export_errors += 1
                if worker_runtime_snapshot and not build_failed:
                    self._worker_runtime_snapshot_export_errors += 1
                if worker_cpu_profile and not build_failed:
                    self._worker_cpu_profile_export_errors += 1
                if terminal_hop_observation and not build_failed:
                    self._terminal_hop_observation_export_errors += 1
                if self._export_errors == 1 or self._export_errors % 100 == 0:
                    logger.warning("Fugue observability export failed: %s", type(exc).__name__)
            finally:
                self._queue.task_done()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> None:
        client = self._client
        if client is None:
            return
        response = await client.post(_endpoint_url(self.config.endpoint or "", path), json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"observability endpoint returned HTTP {response.status_code}")

    def delivery_snapshot(self) -> dict[str, int]:
        queue = self._queue
        return {
            "queue_depth": queue.qsize() if queue is not None else 0,
            "events_dropped_total": self._dropped,
            "export_errors_total": self._export_errors,
            "large_body_decision_enqueued_total": (
                self._large_body_decision_enqueued
            ),
            "large_body_decision_enqueue_dropped_total": (
                self._large_body_decision_enqueue_dropped
            ),
            "large_body_decision_build_errors_total": (
                self._large_body_decision_build_errors
            ),
            "large_body_decision_export_errors_total": (
                self._large_body_decision_export_errors
            ),
            "admission_503_outcome_enqueued_total": (
                self._admission_503_outcome_enqueued
            ),
            "admission_503_outcome_enqueue_dropped_total": (
                self._admission_503_outcome_enqueue_dropped
            ),
            "admission_503_outcome_build_errors_total": (
                self._admission_503_outcome_build_errors
            ),
            "admission_503_outcome_export_errors_total": (
                self._admission_503_outcome_export_errors
            ),
            "response_buffer_event_enqueued_total": (
                self._response_buffer_event_enqueued
            ),
            "response_buffer_event_enqueue_dropped_total": (
                self._response_buffer_event_enqueue_dropped
            ),
            "response_buffer_event_build_errors_total": (
                self._response_buffer_event_build_errors
            ),
            "response_buffer_event_export_errors_total": (
                self._response_buffer_event_export_errors
            ),
            "worker_runtime_snapshot_enqueued_total": (
                self._worker_runtime_snapshot_enqueued
            ),
            "worker_runtime_snapshot_enqueue_dropped_total": (
                self._worker_runtime_snapshot_enqueue_dropped
            ),
            "worker_runtime_snapshot_build_errors_total": (
                self._worker_runtime_snapshot_build_errors
            ),
            "worker_runtime_snapshot_export_errors_total": (
                self._worker_runtime_snapshot_export_errors
            ),
            "worker_cpu_profile_enqueued_total": (
                self._worker_cpu_profile_enqueued
            ),
            "worker_cpu_profile_enqueue_dropped_total": (
                self._worker_cpu_profile_enqueue_dropped
            ),
            "worker_cpu_profile_build_errors_total": (
                self._worker_cpu_profile_build_errors
            ),
            "worker_cpu_profile_export_errors_total": (
                self._worker_cpu_profile_export_errors
            ),
            "terminal_hop_observation_enqueued_total": (
                self._terminal_hop_observation_enqueued
            ),
            "terminal_hop_observation_enqueue_dropped_total": (
                self._terminal_hop_observation_enqueue_dropped
            ),
            "terminal_hop_observation_build_errors_total": (
                self._terminal_hop_observation_build_errors
            ),
            "terminal_hop_observation_export_errors_total": (
                self._terminal_hop_observation_export_errors
            ),
        }


_client: FugueObservabilityClient | None = None


async def start_fugue_observability_from_env(*, service_version: str | None = None) -> None:
    global _client
    config = fugue_observability_config_from_env(service_version=service_version)
    if not config.enabled:
        _client = None
        return
    client = FugueObservabilityClient(config)
    await client.start()
    _client = client


async def stop_fugue_observability() -> None:
    global _client
    client = _client
    _client = None
    if client is not None:
        await client.stop()


def fugue_observability_config_from_env(*, service_version: str | None = None) -> FugueObservabilityConfig:
    endpoint = _env_text("FUGUE_OBSERVABILITY_ENDPOINT") or _env_text("OTEL_EXPORTER_OTLP_ENDPOINT")
    return FugueObservabilityConfig(
        endpoint=endpoint,
        service_name=_env_text("FUGUE_OBSERVABILITY_SERVICE_NAME") or _DEFAULT_SERVICE_NAME,
        service_version=_env_text("FUGUE_OBSERVABILITY_SERVICE_VERSION") or service_version,
        queue_max_size=_env_int("FUGUE_OBSERVABILITY_QUEUE_MAX_SIZE", _DEFAULT_QUEUE_MAX_SIZE),
        export_worker_count=_env_int("FUGUE_OBSERVABILITY_EXPORT_WORKERS", _DEFAULT_EXPORT_WORKER_COUNT),
        export_timeout_seconds=_env_float(
            "FUGUE_OBSERVABILITY_EXPORT_TIMEOUT_SECONDS",
            _DEFAULT_EXPORT_TIMEOUT_SECONDS,
        ),
        sample_rate=max(0.0, min(1.0, _env_float("FUGUE_OBSERVABILITY_SAMPLE_RATE", _DEFAULT_SAMPLE_RATE))),
        identity_attrs=_identity_attrs_from_env(),
        emit_request_summaries=_env_bool("FUGUE_OBSERVABILITY_REQUEST_SUMMARY_ENABLED", True),
        emit_stage_spans=_env_bool("FUGUE_OBSERVABILITY_STAGE_SPANS_ENABLED", True),
        emit_metrics=_env_bool("FUGUE_OBSERVABILITY_METRICS_ENABLED", True),
    )


def emit_uni_api_ember_request_observability(**kwargs: Any) -> None:
    client = _client
    if client is None:
        return
    try:
        client.emit_request(**kwargs)
    except Exception:
        logger.exception("Failed to enqueue Fugue request observability event")


def emit_uni_api_ember_large_body_admission_decision(
    decision: Any,
) -> bool | None:
    client = _client
    if client is None:
        return None
    try:
        return client.emit_large_body_admission_decision(decision)
    except Exception:
        logger.exception("Failed to enqueue Fugue admission decision event")
        return False


def emit_uni_api_ember_admission_503_response_write_outcome(
    outcome: Any,
) -> bool | None:
    client = _client
    if client is None:
        return None
    try:
        return client.emit_admission_503_response_write_outcome(outcome)
    except Exception:
        logger.exception("Failed to enqueue Fugue admission 503 write outcome")
        return False


def emit_uni_api_ember_response_buffer_event(event: Any) -> bool | None:
    client = _client
    if client is None:
        return None
    try:
        return client.emit_response_buffer_event(event)
    except Exception:
        logger.exception("Failed to enqueue Fugue response buffer event")
        return False


def emit_uni_api_ember_worker_runtime_snapshot(
    snapshot: dict[str, Any],
) -> bool | None:
    client = _client
    if client is None:
        return None
    try:
        return client.emit_worker_runtime_snapshot(snapshot)
    except Exception:
        logger.exception("Failed to enqueue Fugue worker runtime snapshot")
        return False


def emit_uni_api_ember_worker_cpu_profile(
    profile: dict[str, Any],
) -> bool | None:
    client = _client
    if client is None:
        return None
    try:
        return client.emit_worker_cpu_profile(profile)
    except Exception:
        logger.exception("Failed to enqueue Fugue worker CPU profile")
        return False


def emit_uni_api_ember_terminal_hop_observation(
    observation: dict[str, Any],
) -> bool | None:
    client = _client
    if client is None:
        return None
    try:
        return client.emit_terminal_hop_observation(observation)
    except Exception:
        logger.exception("Failed to enqueue Fugue terminal hop observation")
        return False


def fugue_observability_delivery_snapshot() -> dict[str, int]:
    client = _client
    return client.delivery_snapshot() if client is not None else {}


def build_uni_api_ember_worker_metric_events(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    timestamp = datetime.now(timezone.utc)
    attributes = _drop_empty(
        {
            **(identity_attrs or {}),
            "component": f"{service_name}-worker",
            "service_version": _safe_text(service_version),
            "metric_scope": "worker_process",
            # Fugue's remote-write processor may intentionally drop this
            # high-cardinality label. It remains available in direct payload
            # inspection and the runtime endpoint; one process is currently
            # enforced per pod.
            "worker_id": _safe_text(snapshot.get("worker_id"), max_len=192),
        }
    )
    values: dict[str, Any] = {
        "uniapi_ember_worker_cpu_seconds_total": snapshot.get(
            "worker_cpu_seconds_total"
        ),
        "uniapi_ember_worker_cpu_cores": snapshot.get("worker_cpu_cores"),
        "uniapi_ember_worker_single_core_saturation_ratio": snapshot.get(
            "worker_single_core_saturation_ratio"
        ),
        "uniapi_ember_worker_sse_events_total": snapshot.get(
            "worker_sse_events_total"
        ),
        "uniapi_ember_worker_sse_bytes_total": snapshot.get(
            "worker_sse_bytes_total"
        ),
        "uniapi_ember_worker_sse_events_per_second": snapshot.get(
            "worker_sse_events_per_second"
        ),
        "uniapi_ember_worker_sse_bytes_per_second": snapshot.get(
            "worker_sse_bytes_per_second"
        ),
        "uniapi_ember_worker_inflight_requests": snapshot.get(
            "worker_inflight_requests"
        ),
        "uniapi_ember_worker_cpu_seconds_per_sse_mebibyte": snapshot.get(
            "worker_cpu_seconds_per_sse_mebibyte"
        ),
        "uniapi_ember_worker_cpu_profile_running": (
            1 if _safe_bool(snapshot.get("worker_cpu_profile_running")) else 0
        ),
        "uniapi_ember_worker_cpu_profile_trigger_total": snapshot.get(
            "worker_cpu_profile_trigger_total"
        ),
        "uniapi_ember_worker_cpu_profile_completed_total": snapshot.get(
            "worker_cpu_profile_completed_total"
        ),
        "uniapi_ember_worker_cpu_profile_failed_total": snapshot.get(
            "worker_cpu_profile_failed_total"
        ),
        "uniapi_ember_oaix_terminal_flush_to_receive_invalid_total": (
            snapshot.get("oaix_terminal_flush_to_ember_receive_invalid_total")
        ),
        "uniapi_ember_oaix_terminal_flush_marker_missing_total": snapshot.get(
            "oaix_terminal_flush_marker_missing_total"
        ),
    }
    histogram = snapshot.get(
        "oaix_terminal_flush_to_ember_receive_histogram"
    )
    if isinstance(histogram, dict):
        values[
            "uniapi_ember_oaix_terminal_flush_to_receive_milliseconds_count"
        ] = histogram.get("count")
        values[
            "uniapi_ember_oaix_terminal_flush_to_receive_milliseconds_sum"
        ] = histogram.get("sum_ms")
        buckets = histogram.get("cumulative_buckets")
        if isinstance(buckets, dict):
            for raw_bound, count in buckets.items():
                bound = _safe_metric_bucket_suffix(raw_bound)
                if bound:
                    values[
                        "uniapi_ember_oaix_terminal_flush_to_receive_"
                        f"milliseconds_bucket_le_{bound}"
                    ] = count
        values[
            "uniapi_ember_oaix_terminal_flush_to_receive_"
            "milliseconds_bucket_le_inf"
        ] = histogram.get("infinite_bucket")

    events: list[dict[str, Any]] = []
    for metric, raw_value in values.items():
        value = _finite_metric_value(raw_value)
        if value is None:
            continue
        events.append(
            {
                "timestamp": _iso_timestamp(timestamp),
                "kind": "metric",
                "source": service_name,
                "message": metric,
                "metric": metric,
                "value": value,
                "attributes": attributes,
            }
        )
    return events


def build_uni_api_ember_worker_runtime_snapshot_event(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    summary = _bounded_worker_runtime_summary(snapshot)
    worker_id = _safe_text(summary.get("worker_id"), max_len=192)
    source_revision = _safe_text(
        summary.get("worker_source_revision"),
        max_len=64,
    )
    summary_json = json.dumps(summary, separators=(",", ":"), sort_keys=True)
    return {
        "timestamp": _iso_timestamp(datetime.now(timezone.utc)),
        "kind": "log",
        "level": "info",
        "service": service_name,
        "source": service_name,
        "event": "worker_runtime_snapshot",
        "event_type": "worker_runtime_snapshot",
        "message": "worker runtime snapshot",
        "app_id": _safe_text((identity_attrs or {}).get("app_id")),
        "attributes": _drop_empty(
            {
                **(identity_attrs or {}),
                "component": f"{service_name}-worker",
                "service_version": _safe_text(service_version),
                "fugue_table": "app_events",
                "severity": "info",
                "worker_id": worker_id,
                "source_revision": source_revision,
                "worker_cpu_cores": _optional_float_text(
                    summary.get("worker_cpu_cores")
                ),
                "worker_inflight_requests": _optional_int_text(
                    summary.get("worker_inflight_requests")
                ),
            }
        ),
        "summary": summary,
        "summary_json": summary_json,
    }


def _bounded_worker_runtime_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "worker_metrics_schema_version": _safe_int(
            snapshot.get("worker_metrics_schema_version"),
            1,
        ),
        "worker_id": _safe_text(snapshot.get("worker_id"), max_len=192),
        "worker_pid": max(0, _safe_int(snapshot.get("worker_pid"), 0)),
        "worker_started_at": _safe_text(
            snapshot.get("worker_started_at"),
            max_len=64,
        ),
        "worker_source_revision": _safe_text(
            snapshot.get("worker_source_revision"),
            max_len=64,
        ),
    }
    numeric_keys = (
        "worker_cpu_seconds_total",
        "worker_cpu_cores",
        "worker_single_core_saturation_ratio",
        "worker_sse_events_total",
        "worker_sse_bytes_total",
        "worker_sse_events_per_second",
        "worker_sse_bytes_per_second",
        "worker_inflight_requests",
        "worker_cpu_seconds_per_sse_mebibyte",
        "worker_metrics_sample_elapsed_seconds",
        "worker_cpu_profile_trigger_cores",
        "worker_cpu_profile_trigger_samples",
        "worker_cpu_profile_trigger_streak",
        "worker_cpu_profile_trigger_total",
        "worker_cpu_profile_completed_total",
        "worker_cpu_profile_failed_total",
        "worker_phase_sample_rate",
        "worker_socket_unread_samples_total",
        "worker_socket_unread_bytes_total",
        "worker_socket_unread_bytes_max",
        "worker_socket_unread_bytes_last",
        "worker_socket_unread_sample_failures_total",
        "oaix_terminal_flush_to_ember_receive_invalid_total",
        "oaix_terminal_flush_marker_missing_total",
    )
    for key in numeric_keys:
        value = _finite_metric_value(snapshot.get(key))
        if value is not None:
            summary[key] = value
    summary["worker_cpu_profile_enabled"] = _safe_bool(
        snapshot.get("worker_cpu_profile_enabled")
    )
    summary["worker_cpu_profile_running"] = _safe_bool(
        snapshot.get("worker_cpu_profile_running")
    )

    raw_phase_samples = snapshot.get("worker_phase_samples")
    if isinstance(raw_phase_samples, dict):
        phase_samples: dict[str, dict[str, float | int]] = {}
        for phase in _WORKER_PERFORMANCE_PHASES:
            raw_metrics = raw_phase_samples.get(phase)
            if not isinstance(raw_metrics, dict):
                continue
            metrics: dict[str, float | int] = {}
            for key in (
                "samples_total",
                "wall_ns_total",
                "cpu_ns_total",
                "bytes_total",
                "events_total",
                "wall_us_per_event",
                "cpu_us_per_event",
            ):
                value = _finite_metric_value(raw_metrics.get(key))
                if value is not None and value >= 0:
                    metrics[key] = value
            if metrics:
                phase_samples[phase] = metrics
        if phase_samples:
            summary["worker_phase_samples"] = phase_samples

    raw_phase_sampling = snapshot.get("worker_phase_sampling")
    if isinstance(raw_phase_sampling, dict):
        phase_sampling: dict[str, dict[str, float | int]] = {}
        for scope in _WORKER_PHASE_SAMPLE_SCOPES:
            raw_metrics = raw_phase_sampling.get(scope)
            if not isinstance(raw_metrics, dict):
                continue
            metrics: dict[str, float | int] = {}
            for key in ("candidates_total", "selected_total"):
                value = _finite_metric_value(raw_metrics.get(key))
                if value is not None and value >= 0:
                    metrics[key] = value
            if metrics:
                phase_sampling[scope] = metrics
        if phase_sampling:
            summary["worker_phase_sampling"] = phase_sampling

    raw_threadpool = snapshot.get("worker_threadpool_tasks")
    if isinstance(raw_threadpool, dict):
        raw_categories = raw_threadpool.get("categories")
        categories: dict[str, dict[str, float | int]] = {}
        if isinstance(raw_categories, dict):
            for category in _WORKER_THREADPOOL_TASK_CATEGORIES:
                raw_metrics = raw_categories.get(category)
                if not isinstance(raw_metrics, dict):
                    continue
                metrics: dict[str, float | int] = {}
                for key in (
                    "submitted_total",
                    "started_total",
                    "completed_total",
                    "failed_total",
                    "cancelled_total",
                    "cancelled_task_started_total",
                    "queued",
                    "inflight",
                    "active_threads",
                    "queue_wait_ns_total",
                    "wall_ns_total",
                    "cpu_ns_total",
                ):
                    value = _finite_metric_value(raw_metrics.get(key))
                    if value is not None and value >= 0:
                        metrics[key] = value
                if metrics:
                    categories[category] = metrics
        raw_dedicated = raw_threadpool.get("dedicated_executors")
        dedicated: dict[str, dict[str, float | int]] = {}
        if isinstance(raw_dedicated, dict):
            for category in _WORKER_THREADPOOL_TASK_CATEGORIES:
                raw_metrics = raw_dedicated.get(category)
                if not isinstance(raw_metrics, dict):
                    continue
                metrics: dict[str, float | int] = {}
                for key in ("queue_depth", "threads", "alive_threads"):
                    value = _finite_metric_value(raw_metrics.get(key))
                    if value is not None and value >= 0:
                        metrics[key] = value
                if metrics:
                    dedicated[category] = metrics
        if categories or dedicated:
            threadpool_summary: dict[str, Any] = {
                "schema_version": max(
                    0,
                    _safe_int(raw_threadpool.get("schema_version"), 0),
                ),
                "categories": categories,
                "dedicated_executors": dedicated,
            }
            if (
                raw_threadpool.get("lifecycle_semantics")
                == _WORKER_THREADPOOL_LIFECYCLE_SEMANTICS
            ):
                threadpool_summary["lifecycle_semantics"] = (
                    _WORKER_THREADPOOL_LIFECYCLE_SEMANTICS
                )
            summary["worker_threadpool_tasks"] = threadpool_summary

    histogram = snapshot.get("oaix_terminal_flush_to_ember_receive_histogram")
    if isinstance(histogram, dict):
        safe_histogram: dict[str, Any] = {}
        for source_key, target_key in (
            ("count", "count"),
            ("sum_ms", "sum_ms"),
            ("infinite_bucket", "infinite_bucket"),
        ):
            value = _finite_metric_value(histogram.get(source_key))
            if value is not None and value >= 0:
                safe_histogram[target_key] = value
        buckets = histogram.get("cumulative_buckets")
        if isinstance(buckets, dict):
            safe_buckets: dict[str, float | int] = {}
            for raw_bound, raw_count in list(buckets.items())[:32]:
                bound = _safe_metric_bucket_suffix(raw_bound)
                count = _finite_metric_value(raw_count)
                if bound and count is not None and count >= 0:
                    safe_buckets[bound.replace("_", ".")] = count
            safe_histogram["cumulative_buckets"] = safe_buckets
        summary["oaix_terminal_flush_to_ember_receive_histogram"] = (
            safe_histogram
        )
    return {
        key: value
        for key, value in summary.items()
        if value is not None and value != ""
    }


def build_uni_api_ember_terminal_hop_metric_event(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    observation: dict[str, Any],
) -> dict[str, Any]:
    lag_ms = _finite_metric_value(observation.get("lag_ms"))
    if lag_ms is None or lag_ms < 0:
        raise ValueError("terminal hop lag must be a finite non-negative value")
    timestamp = _parse_observation_timestamp(
        observation.get("terminal_received_at")
    )
    metric = "uniapi_ember_oaix_terminal_flush_to_receive_milliseconds"
    return {
        "timestamp": _iso_timestamp(timestamp),
        "kind": "metric",
        "source": service_name,
        "message": metric,
        "metric": metric,
        "value": lag_ms,
        "attributes": _drop_empty(
            {
                **(identity_attrs or {}),
                "component": f"{service_name}-worker",
                "service_version": _safe_text(service_version),
                "metric_scope": "request_hop",
            }
        ),
    }


def build_uni_api_ember_worker_cpu_profile_event(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    profile: dict[str, Any],
) -> dict[str, Any]:
    summary = _bounded_cpu_profile_summary(profile)
    status = _safe_text(summary.get("status"), max_len=32) or "unknown"
    level = "info" if status == "completed" else "warning"
    summary_json = json.dumps(summary, separators=(",", ":"), sort_keys=True)
    return {
        "timestamp": _safe_text(summary.get("finished_at"))
        or _iso_timestamp(datetime.now(timezone.utc)),
        "kind": "log",
        "level": level,
        "service": service_name,
        "source": service_name,
        "event": "worker_on_cpu_profile",
        "event_type": "worker_on_cpu_profile",
        "message": f"worker on-CPU profile {status}",
        "app_id": _safe_text((identity_attrs or {}).get("app_id")),
        "attributes": _drop_empty(
            {
                **(identity_attrs or {}),
                "component": f"{service_name}-worker",
                "service_version": _safe_text(service_version),
                "fugue_table": "app_events",
                "severity": level,
                "profile_id": _safe_text(summary.get("profile_id")),
                "worker_id": _safe_text(summary.get("worker_id")),
                "source_revision": _safe_text(summary.get("source_revision")),
                "profile_status": status,
                "trigger_cpu_cores": _optional_float_text(
                    summary.get("trigger_cpu_cores")
                ),
                "profiled_cpu_seconds": _optional_float_text(
                    summary.get("profiled_cpu_seconds")
                ),
                "sample_rounds": _optional_int_text(
                    summary.get("sample_rounds")
                ),
            }
        ),
        "summary": summary,
        "summary_json": summary_json,
    }


def _bounded_cpu_profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": _safe_int(profile.get("schema_version"), 1),
        "profile_id": _safe_text(profile.get("profile_id"), max_len=64),
        "worker_id": _safe_text(profile.get("worker_id"), max_len=192),
        "source_revision": _safe_text(
            profile.get("source_revision"), max_len=64
        ),
        "status": _safe_text(profile.get("status"), max_len=32),
        "error_type": _safe_text(profile.get("error_type"), max_len=96),
        "started_at": _safe_text(profile.get("started_at"), max_len=64),
        "finished_at": _safe_text(profile.get("finished_at"), max_len=64),
    }
    if (
        profile.get("threadpool_classification_semantics")
        == _WORKER_THREADPOOL_PROFILE_SEMANTICS
    ):
        summary["threadpool_classification_semantics"] = (
            _WORKER_THREADPOOL_PROFILE_SEMANTICS
        )
    for key in (
        "trigger_cpu_cores",
        "configured_duration_seconds",
        "observed_duration_seconds",
        "sample_hz",
        "profiled_cpu_seconds",
    ):
        value = _finite_metric_value(profile.get(key))
        if value is not None:
            summary[key] = value
    for key in (
        "sample_rounds",
        "active_thread_samples",
        "profiled_cpu_ticks",
        "proc_read_errors",
    ):
        if profile.get(key) is not None:
            summary[key] = max(0, _safe_int(profile.get(key), 0))

    leaf_rows = []
    raw_leaves = profile.get("top_leaf_functions")
    if isinstance(raw_leaves, list):
        for raw in raw_leaves[:20]:
            if not isinstance(raw, dict):
                continue
            leaf_rows.append(
                _drop_empty(
                    {
                        "function": _safe_text(
                            raw.get("function"), max_len=320
                        ),
                        "cpu_ticks": max(0, _safe_int(raw.get("cpu_ticks"), 0)),
                        "cpu_seconds": _finite_metric_value(
                            raw.get("cpu_seconds")
                        ),
                    }
                )
            )
    summary["top_leaf_functions"] = leaf_rows

    stack_rows = []
    raw_stacks = profile.get("top_stacks")
    if isinstance(raw_stacks, list):
        for raw in raw_stacks[:20]:
            if not isinstance(raw, dict):
                continue
            stack = raw.get("stack")
            safe_stack = []
            if isinstance(stack, list):
                safe_stack = [
                    text
                    for value in stack[:24]
                    if (text := _safe_text(value, max_len=320))
                ]
            stack_rows.append(
                {
                    "stack": safe_stack,
                    "cpu_ticks": max(0, _safe_int(raw.get("cpu_ticks"), 0)),
                    "cpu_seconds": _finite_metric_value(raw.get("cpu_seconds")),
                    "samples": max(0, _safe_int(raw.get("samples"), 0)),
                }
            )
    summary["top_stacks"] = stack_rows

    category_rows = []
    raw_categories = profile.get("threadpool_categories")
    if isinstance(raw_categories, list):
        allowed_categories = set(_WORKER_THREADPOOL_TASK_CATEGORIES)
        for raw in raw_categories[: len(allowed_categories)]:
            if not isinstance(raw, dict):
                continue
            category = _safe_text(raw.get("category"), max_len=48)
            if category not in allowed_categories:
                continue
            category_row: dict[str, Any] = _drop_empty(
                {
                    "category": category,
                    "cpu_ticks": max(
                        0,
                        _safe_int(raw.get("cpu_ticks"), 0),
                    ),
                    "cpu_seconds": _finite_metric_value(
                        raw.get("cpu_seconds")
                    ),
                    "samples": max(
                        0,
                        _safe_int(raw.get("samples"), 0),
                    ),
                }
            )
            raw_sources = raw.get("sources")
            sources = []
            if isinstance(raw_sources, list):
                for raw_source in raw_sources[:3]:
                    if not isinstance(raw_source, dict):
                        continue
                    source = _safe_text(
                        raw_source.get("source"),
                        max_len=48,
                    )
                    if source not in _WORKER_THREADPOOL_CATEGORY_SOURCES:
                        continue
                    sources.append(
                        _drop_empty(
                            {
                                "source": source,
                                "cpu_ticks": max(
                                    0,
                                    _safe_int(
                                        raw_source.get("cpu_ticks"),
                                        0,
                                    ),
                                ),
                                "cpu_seconds": _finite_metric_value(
                                    raw_source.get("cpu_seconds")
                                ),
                                "samples": max(
                                    0,
                                    _safe_int(
                                        raw_source.get("samples"),
                                        0,
                                    ),
                                ),
                            }
                        )
                    )
            if sources:
                category_row["sources"] = sources
            category_rows.append(category_row)
    summary["threadpool_categories"] = category_rows
    return {
        key: value
        for key, value in summary.items()
        if value is not None and value != ""
    }


def _finite_metric_value(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) <= (1 << 53):
        return int(number)
    return number


def _optional_float_text(value: Any) -> str | None:
    number = _finite_metric_value(value)
    return None if number is None else str(number)


def _safe_metric_bucket_suffix(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace(".", "_")
    if not text or len(text) > 24:
        return None
    if not all(character.isdigit() or character == "_" for character in text):
        return None
    return text


def _parse_observation_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def build_uni_api_ember_response_buffer_event(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    response_event: Any,
) -> dict[str, Any]:
    raw = asdict(response_event)
    occurred_at_ms = _safe_int(raw.get("occurred_at_unix_ms"), 0)
    observed_at = datetime.fromtimestamp(
        max(0, occurred_at_ms) / 1000.0,
        tz=timezone.utc,
    )
    rejected = raw.get("event") == "reject"
    attributes = _drop_empty(
        {
            **(identity_attrs or {}),
            **raw,
            "component": service_name,
            "service_version": service_version,
            "event_type": "response_buffer_lifecycle",
            "fugue_table": "app_events",
            "severity": "warning" if rejected else "info",
        }
    )
    summary_json = json.dumps(raw, separators=(",", ":"), sort_keys=True)
    return {
        "timestamp": _iso_timestamp(observed_at),
        "kind": "log",
        "level": attributes["severity"],
        "service": service_name,
        "source": service_name,
        "event": "response_buffer_lifecycle",
        "event_type": "response_buffer_lifecycle",
        "message": (
            f"response buffer {raw.get('event', 'event')} "
            f"{raw.get('outcome', '')}"
        ).strip(),
        "app_id": _safe_text((identity_attrs or {}).get("app_id")),
        "trace_id": _safe_text(raw.get("request_self_trace_id")),
        "request_id": _safe_text(raw.get("request_self_request_id")),
        "attributes": attributes,
        "summary": raw,
        "summary_json": summary_json,
    }


def build_uni_api_ember_admission_503_response_write_event(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    outcome: Any,
) -> dict[str, Any]:
    raw = asdict(outcome)
    occurred_at_ms = _safe_int(raw.get("occurred_at_unix_ms"), 0)
    observed_at = datetime.fromtimestamp(
        max(0, occurred_at_ms) / 1000.0,
        tz=timezone.utc,
    )
    completed = _safe_bool(raw.get("asgi_response_write_completed"))
    attributes = _drop_empty(
        {
            **(identity_attrs or {}),
            **raw,
            "component": service_name,
            "service_version": service_version,
            "event_type": "admission_503_response_write_outcome",
            "fugue_table": "app_events",
            "severity": "info" if completed else "warning",
        }
    )
    return {
        "timestamp": _iso_timestamp(observed_at),
        "kind": "log",
        "level": attributes["severity"],
        "service": service_name,
        "source": service_name,
        "event": "admission_503_response_write_outcome",
        "event_type": "admission_503_response_write_outcome",
        "message": (
            "admission 503 ASGI response write completed"
            if completed
            else "admission 503 ASGI response write failed"
        ),
        "app_id": _safe_text((identity_attrs or {}).get("app_id")),
        "trace_id": _safe_text(raw.get("request_self_trace_id")),
        "request_id": _safe_text(raw.get("request_self_request_id")),
        "path": _safe_text(raw.get("request_self_path")),
        "status_code": 503,
        "attributes": attributes,
        "summary": raw,
        "summary_json": json.dumps(raw, separators=(",", ":"), sort_keys=True),
    }


def build_uni_api_ember_large_body_admission_event(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    decision: Any,
) -> dict[str, Any]:
    raw = asdict(decision)
    occurred_at_ms = _safe_int(raw.get("occurred_at_unix_ms"), 0)
    observed_at = datetime.fromtimestamp(
        max(0, occurred_at_ms) / 1000.0,
        tz=timezone.utc,
    )
    holder = raw.get("holder") if isinstance(raw.get("holder"), dict) else {}
    blocking_holders = raw.get("blocking_holders")
    blocking_holder_count = (
        len(blocking_holders) if isinstance(blocking_holders, (list, tuple)) else 0
    )
    scalar_summary = {
        key: value
        for key, value in raw.items()
        if not isinstance(value, (dict, list, tuple))
    }
    attributes = _drop_empty(
        {
            **(identity_attrs or {}),
            **scalar_summary,
            "component": service_name,
            "service_version": service_version,
            "event_type": "large_body_admission_decision",
            "fugue_table": "app_events",
            "severity": (
                "warning" if raw.get("decision") == "reject" else "info"
            ),
            "blocking_holder_count": blocking_holder_count,
            "holder_claim_id": holder.get("claim_id"),
            "holder_lease_id": holder.get("lease_id"),
            "holder_request_id": holder.get("request_id"),
            "holder_trace_id": holder.get("trace_id"),
            "holder_claimed_at_unix_ms": holder.get("claimed_at_unix_ms"),
            "holder_held_ms": holder.get("held_ms"),
        }
    )
    summary_json = json.dumps(raw, separators=(",", ":"), sort_keys=True)
    return {
        "timestamp": _iso_timestamp(observed_at),
        "kind": "log",
        "level": attributes.get("severity", "info"),
        "service": service_name,
        "source": service_name,
        "event": "large_body_admission_decision",
        "event_type": "large_body_admission_decision",
        "message": f"large body admission {raw.get('decision', 'decision')}",
        "app_id": _safe_text((identity_attrs or {}).get("app_id")),
        "trace_id": _safe_text(raw.get("request_self_trace_id")),
        "request_id": _safe_text(raw.get("request_self_request_id")),
        "path": _safe_text(raw.get("request_self_path")),
        "attributes": attributes,
        "summary": raw,
        "summary_json": summary_json,
    }


def build_uni_api_ember_request_telemetry(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    current_info: dict[str, Any],
    runtime_metrics: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    now = datetime.now(timezone.utc)
    spans = dict(current_info.get("timing_spans") or {})
    trace_id = _safe_text(current_info.get("trace_id") or spans.get("trace_id"))
    request_id = _safe_text(current_info.get("request_id"))
    endpoint = _safe_text(current_info.get("endpoint"))
    method, path_template = _split_endpoint(endpoint)
    status_code = _safe_int(current_info.get("status_code"), 0)
    route_id = _route_id(endpoint)
    duration_ms = _duration_ms_from_info(current_info)
    ttft_ms = _ttft_ms(spans)
    error_type = _safe_text(current_info.get("error_type")) or _classify_error(status_code)
    stream_outcome = _safe_text(current_info.get("stream_outcome"))
    stream_error_status_code = _safe_int(
        current_info.get("stream_error_status_code"), 0
    )
    retry_count = _safe_int(current_info.get("retry_count"), 0)
    attempt_count = _safe_int(current_info.get("attempt_count"), 0)
    retry_decision_count = _safe_int(
        current_info.get("retry_decision_count"),
        retry_count,
    )
    retry_transition_count = _safe_int(
        current_info.get("retry_transition_count"),
        0,
    )
    planned_attempt_count = _safe_int(
        current_info.get("planned_attempt_count"),
        0,
    )
    transport_error_count = _safe_int(
        current_info.get("transport_error_count"),
        0,
    )
    local_overload_count = _safe_int(
        current_info.get("local_overload_count"),
        0,
    )
    cooldown_count = _safe_int(current_info.get("cooldown_count"), 0)
    is_stream = _safe_bool(current_info.get("stream"))
    api_key_hash = _secret_hash(current_info.get("api_key"))
    responses_diagnostics = _responses_stream_diagnostics(current_info)
    image_stream_diagnostics = current_info.get("image_stream_diagnostics")
    if not isinstance(image_stream_diagnostics, dict):
        image_stream_diagnostics = {}
    transport_attempt: dict[str, Any] = {}
    routing_attempts = current_info.get("routing_attempts")
    if isinstance(routing_attempts, list):
        for candidate in reversed(routing_attempts):
            if not isinstance(candidate, dict):
                continue
            if any(key in candidate for key in _TRANSPORT_PHASE_FIELDS):
                transport_attempt = candidate
                break
    transport_phase_attrs = _transport_phase_attrs(transport_attempt)
    request_body_complexity_attrs = _request_body_complexity_attrs(
        current_info
    )
    runtime_snapshot = runtime_metrics if isinstance(runtime_metrics, dict) else {}
    outbound_network_resources = runtime_snapshot.get(
        "outbound_network_resources"
    )
    if not isinstance(outbound_network_resources, dict):
        outbound_network_resources = {}

    base = _base_attrs(
        service_name=service_name,
        service_version=service_version,
        identity_attrs=identity_attrs,
        trace_id=trace_id,
        request_id=request_id,
        parent_span_id=_safe_text(current_info.get("parent_span_id") or spans.get("parent_span_id")),
        endpoint=endpoint,
        method=method,
        path_template=path_template,
        route_id=route_id,
        model=_safe_text(current_info.get("model")),
        provider=_safe_text(current_info.get("provider")),
        role=_safe_text(current_info.get("role")),
        is_stream=is_stream,
        status_code=status_code,
        error_type=error_type,
        retry_count=retry_count,
        cooldown_count=cooldown_count,
        api_key_hash=api_key_hash,
    )
    base.update(
        _drop_empty(
            {
                "attempt_count": _int_text(attempt_count),
                "retry_decision_count": _int_text(retry_decision_count),
                "retry_transition_count": _int_text(retry_transition_count),
                "planned_attempt_count": _int_text(planned_attempt_count),
                "transport_error_count": _int_text(transport_error_count),
                "local_overload_count": _int_text(local_overload_count),
            }
        )
    )

    logs = [
        {
            "timestamp": _iso_timestamp(now),
            "level": _event_level(stream_error_status_code or status_code),
            "service": service_name,
            "trace_id": trace_id,
            "request_id": request_id,
            "event": "request_summary",
            "event_type": "request_summary",
            "source": service_name,
            "message": "uni-api-ember request finished",
            # Fugue request_facts indexes these fields at the event top level.
            # Keep the nested attributes too for existing log consumers.
            "app_id": _safe_text((identity_attrs or {}).get("app_id")),
            "path": path_template or endpoint,
            "status_code": status_code,
            "attributes": _drop_empty(
                {
                    **base,
                    "duration_ms": _int_text(duration_ms),
                    "total_ms": _int_text(duration_ms),
                    "ttfb_ms": _int_text(ttft_ms),
                    "ttft_ms": _int_text(ttft_ms),
                    "upstream_ms": _int_text(_stage_delta_ms(spans, "upstream_headers_received", "upstream_send_start")),
                    "status_class": _status_class(status_code),
                    "status_origin": _safe_text(
                        current_info.get("status_origin"),
                        max_len=64,
                    ),
                    "request_kind": _safe_text(current_info.get("request_kind")),
                    "stream_outcome": stream_outcome,
                    "stream_error_status_code": _optional_int_text(
                        stream_error_status_code
                    ),
                    "stream_error_code": _safe_text(
                        current_info.get("stream_error_code"),
                        max_len=128,
                    ),
                    "stream_error_type": _safe_text(
                        current_info.get("stream_error_type"),
                        max_len=128,
                    ),
                    "stream_error_event_type": _safe_text(
                        current_info.get("stream_error_event_type"),
                        max_len=128,
                    ),
                    "stream_error_after_response_start": _bool_text(
                        _safe_bool(
                            current_info.get("stream_error_after_response_start")
                        )
                    ),
                    "downstream_disconnected": _bool_text(
                        _safe_bool(current_info.get("downstream_disconnected"))
                    ),
                    "postcommit_sse_protocol_error_isolated": _bool_text(
                        _safe_bool(
                            current_info.get(
                                "postcommit_sse_protocol_error_isolated"
                            )
                        )
                    ),
                    "image_stream_contract_version": _optional_int_text(
                        image_stream_diagnostics.get("contract_version")
                    ),
                    "image_stream_last_event_type": _safe_text(
                        image_stream_diagnostics.get("last_event_type"),
                        max_len=96,
                    ),
                    "image_stream_last_data_type": _safe_text(
                        image_stream_diagnostics.get("last_data_type"),
                        max_len=96,
                    ),
                    "image_stream_eof": _bool_text(
                        _safe_bool(image_stream_diagnostics.get("eof"))
                    ),
                    "image_stream_terminal_seen": _bool_text(
                        _safe_bool(
                            image_stream_diagnostics.get("terminal_seen")
                        )
                    ),
                    "image_stream_synthetic_terminal": _bool_text(
                        _safe_bool(
                            image_stream_diagnostics.get(
                                "synthetic_terminal"
                            )
                        )
                    ),
                    "image_stream_synthetic_terminal_type": _safe_text(
                        image_stream_diagnostics.get(
                            "synthetic_terminal_type"
                        ),
                        max_len=96,
                    ),
                    **request_body_complexity_attrs,
                    **transport_phase_attrs,
                }
            ),
            "summary": _drop_empty(
                {
                    "message_roles": _safe_text(current_info.get("message_roles")),
                    "role_counts": _safe_text(current_info.get("role_counts")),
                    "attempt_count": _int_text(attempt_count),
                    "status_origin": _safe_text(
                        current_info.get("status_origin"),
                        max_len=64,
                    ),
                    "retry_decision_count": _int_text(retry_decision_count),
                    "retry_transition_count": _int_text(
                        retry_transition_count
                    ),
                    "planned_attempt_count": _int_text(planned_attempt_count),
                    "planned_retry_count": _int_text(
                        _safe_int(current_info.get("planned_retry_count"), 0)
                    ),
                    "matching_provider_count": _int_text(
                        _safe_int(current_info.get("matching_provider_count"), 0)
                    ),
                    "routing_attempts_omitted_count": _int_text(
                        _safe_int(
                            current_info.get("routing_attempts_omitted_count"),
                            0,
                        )
                    ),
                    "transport_error_count": _int_text(
                        transport_error_count
                    ),
                    "local_overload_count": _int_text(
                        local_overload_count
                    ),
                    **transport_phase_attrs,
                    "client_pool_wait_ms": _int_text(_span_ms(spans, "upstream_pool_wait_ms")),
                    "request_admission_wait_ms": _int_text(
                        _span_ms(spans, "request_admission_wait_ms")
                    ),
                    "event_loop_lag_ms": _int_text(_runtime_int(runtime_metrics, "event_loop_lag_ms")),
                    "inflight_requests": _int_text(_runtime_int(runtime_metrics, "inflight_requests")),
                    "request_waiters": _int_text(_runtime_int(runtime_metrics, "request_waiters")),
                    "request_admission_mode": _safe_text(
                        runtime_snapshot.get("request_admission_mode")
                    ),
                    "request_control_memory_reserved_bytes": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "request_control_memory_reserved_bytes",
                        )
                    ),
                    "cpu_phase_capacity": _int_text(
                        _runtime_int(runtime_metrics, "cpu_phase_capacity")
                    ),
                    "cpu_phase_active": _int_text(
                        _runtime_int(runtime_metrics, "cpu_phase_active")
                    ),
                    "cpu_phase_waiters": _int_text(
                        _runtime_int(runtime_metrics, "cpu_phase_waiters")
                    ),
                    "runtime_global_large_body_active": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_active",
                        )
                    ),
                    "runtime_global_large_body_limit": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_limit",
                        )
                    ),
                    "runtime_global_large_body_threshold_weighted_bytes": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_threshold_weighted_bytes",
                        )
                    ),
                    "runtime_global_large_body_decision_record_failures_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_decision_record_failures_total",
                        )
                    ),
                    "runtime_global_large_body_decision_observer_errors_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_decision_observer_errors_total",
                        )
                    ),
                    "runtime_global_large_body_decision_observer_enqueue_failures_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_decision_observer_enqueue_failures_total",
                        )
                    ),
                    "runtime_global_large_body_decision_export_enqueued_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_decision_export_enqueued_total",
                        )
                    ),
                    "runtime_global_large_body_decision_export_enqueue_dropped_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_decision_export_enqueue_dropped_total",
                        )
                    ),
                    "runtime_global_large_body_decision_export_build_errors_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_decision_export_build_errors_total",
                        )
                    ),
                    "runtime_global_large_body_decision_export_errors_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_large_body_decision_export_errors_total",
                        )
                    ),
                    "runtime_global_admission_503_outcome_export_enqueued_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_admission_503_outcome_export_enqueued_total",
                        )
                    ),
                    "runtime_global_admission_503_outcome_export_enqueue_dropped_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_admission_503_outcome_export_enqueue_dropped_total",
                        )
                    ),
                    "runtime_global_admission_503_outcome_export_build_errors_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_admission_503_outcome_export_build_errors_total",
                        )
                    ),
                    "runtime_global_admission_503_outcome_export_errors_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_admission_503_outcome_export_errors_total",
                        )
                    ),
                    "runtime_global_admission_rejection_decision_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_admission_rejection_decision_total",
                        )
                    ),
                    "runtime_global_admission_503_response_write_completed_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_admission_503_response_write_completed_total",
                        )
                    ),
                    "runtime_global_admission_503_response_write_failed_total": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_admission_503_response_write_failed_total",
                        )
                    ),
                    "runtime_global_request_body_reserved_weighted_bytes": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_request_body_reserved_weighted_bytes",
                        )
                    ),
                    "runtime_global_upstream_response_reserved_weighted_bytes": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_upstream_response_reserved_weighted_bytes",
                        )
                    ),
                    "runtime_global_retained_reserved_weighted_bytes": _int_text(
                        _runtime_int(
                            runtime_metrics,
                            "runtime_global_retained_reserved_weighted_bytes",
                        )
                    ),
                    "waiting_first_byte": _int_text(_runtime_int(runtime_metrics, "waiting_first_byte")),
                    **request_body_complexity_attrs,
                    "upstream_pool_in_use": _int_text(
                        _runtime_int(runtime_metrics, "upstream_pool_in_use")
                    ),
                    "upstream_pool_waiters": _int_text(
                        _runtime_int(runtime_metrics, "upstream_pool_waiters")
                    ),
                    "outbound_fd_headroom": _int_text(
                        _safe_int(
                            outbound_network_resources.get("fd_headroom"),
                            0,
                        )
                    ),
                    "outbound_ephemeral_port_headroom": _int_text(
                        _safe_int(
                            outbound_network_resources.get(
                                "ephemeral_port_headroom"
                            ),
                            0,
                        )
                    ),
                    "stream_queue_bytes": _int_text(
                        _runtime_int(runtime_metrics, "stream_queue_bytes")
                    ),
                    "stream_queue_peak_bytes": _int_text(
                        _safe_int(current_info.get("stream_queue_peak_bytes"), 0)
                    ),
                    "responses_delta_fast_path_candidates": _optional_int_text(
                        current_info.get("responses_delta_fast_path_candidates")
                    ),
                    "responses_delta_fast_path_events": _optional_int_text(
                        current_info.get("responses_delta_fast_path_events")
                    ),
                    "responses_delta_fast_path_fallbacks": _optional_int_text(
                        current_info.get("responses_delta_fast_path_fallbacks")
                    ),
                    "responses_delta_fast_path_bytes": _optional_int_text(
                        current_info.get("responses_delta_fast_path_bytes")
                    ),
                    "rust_responses_data_plane": _bool_text(
                        _safe_bool(
                            current_info.get("rust_responses_data_plane")
                        )
                    ),
                    "rust_responses_control_version": _optional_int_text(
                        current_info.get("rust_responses_control_version")
                    ),
                    "rust_responses_external_committed": _bool_text(
                        _safe_bool(
                            current_info.get(
                                "rust_responses_external_committed"
                            )
                        )
                    ),
                    "rust_responses_commit_reason": _safe_text(
                        current_info.get("rust_responses_commit_reason"),
                        max_len=80,
                    ),
                    "rust_responses_precommit_events": _optional_int_text(
                        current_info.get("rust_responses_precommit_events")
                    ),
                    "rust_responses_precommit_bytes": _optional_int_text(
                        current_info.get("rust_responses_precommit_bytes")
                    ),
                    "rust_responses_upstream_bytes": _optional_int_text(
                        current_info.get("rust_responses_upstream_bytes")
                    ),
                    "rust_responses_downstream_bytes": _optional_int_text(
                        current_info.get("rust_responses_downstream_bytes")
                    ),
                    "rust_responses_sse_events": _optional_int_text(
                        current_info.get("rust_responses_sse_events")
                    ),
                    "rust_responses_delta_events": _optional_int_text(
                        current_info.get("rust_responses_delta_events")
                    ),
                    "rust_responses_normalized_events": _optional_int_text(
                        current_info.get("rust_responses_normalized_events")
                    ),
                    "rust_request_spool": _bool_text(
                        _safe_bool(current_info.get("rust_request_spool"))
                    ),
                    "rust_request_spool_body_bytes": _optional_int_text(
                        current_info.get("rust_request_spool_body_bytes")
                    ),
                    "rust_request_spool_memory_peak_bytes": _optional_int_text(
                        current_info.get("rust_request_spool_memory_peak_bytes")
                    ),
                    "rust_request_spool_local_disk_bytes": _optional_int_text(
                        current_info.get("rust_request_spool_local_disk_bytes")
                    ),
                    "rust_request_spool_local_free_bytes_start": _optional_int_text(
                        current_info.get("rust_request_spool_local_free_bytes_start")
                    ),
                    "rust_request_spool_local_writable_bytes_start": _optional_int_text(
                        current_info.get("rust_request_spool_local_writable_bytes_start")
                    ),
                    "rust_request_spool_local_free_inodes_start": _optional_int_text(
                        current_info.get("rust_request_spool_local_free_inodes_start")
                    ),
                    "rust_request_spool_local_writable_inodes_start": _optional_int_text(
                        current_info.get("rust_request_spool_local_writable_inodes_start")
                    ),
                    "rust_request_spool_resource_wait_ms": _optional_int_text(
                        current_info.get("rust_request_spool_resource_wait_ms")
                    ),
                    "rust_request_spool_final_tier": _safe_text(
                        current_info.get("rust_request_spool_final_tier"),
                        max_len=32,
                    ),
                    "rust_request_spool_failure_resource": _safe_text(
                        current_info.get("rust_request_spool_failure_resource"),
                        max_len=80,
                    ),
                    "wire_status_code": _optional_int_text(
                        current_info.get("wire_status_code")
                    ),
                    "response_committed": _bool_text(
                        _safe_bool(current_info.get("response_committed"))
                    ),
                    "postcommit_sse_protocol_error_isolated": _bool_text(
                        _safe_bool(
                            current_info.get(
                                "postcommit_sse_protocol_error_isolated"
                            )
                        )
                    ),
                    "image_stream_contract_version": _optional_int_text(
                        image_stream_diagnostics.get("contract_version")
                    ),
                    "image_stream_last_event_type": _safe_text(
                        image_stream_diagnostics.get("last_event_type"),
                        max_len=96,
                    ),
                    "image_stream_last_data_type": _safe_text(
                        image_stream_diagnostics.get("last_data_type"),
                        max_len=96,
                    ),
                    "image_stream_eof": _bool_text(
                        _safe_bool(image_stream_diagnostics.get("eof"))
                    ),
                    "image_stream_terminal_seen": _bool_text(
                        _safe_bool(
                            image_stream_diagnostics.get("terminal_seen")
                        )
                    ),
                    "image_stream_synthetic_terminal": _bool_text(
                        _safe_bool(
                            image_stream_diagnostics.get(
                                "synthetic_terminal"
                            )
                        )
                    ),
                    "image_stream_synthetic_terminal_type": _safe_text(
                        image_stream_diagnostics.get(
                            "synthetic_terminal_type"
                        ),
                        max_len=96,
                    ),
                    "prompt_tokens": _responses_token_text(
                        current_info,
                        responses_diagnostics,
                        value_key="prompt_tokens",
                        known_key="downstream_usage_input_known",
                    ),
                    "completion_tokens": _responses_token_text(
                        current_info,
                        responses_diagnostics,
                        value_key="completion_tokens",
                        known_key="downstream_usage_output_known",
                    ),
                    "total_tokens": _responses_token_text(
                        current_info,
                        responses_diagnostics,
                        value_key="total_tokens",
                        known_key="downstream_usage_total_known",
                    ),
                    "usage_parse_error": _safe_text(
                        current_info.get("usage_parse_error"), max_len=80
                    ),
                    "semantic_status": _safe_text(
                        responses_diagnostics.get("semantic_status")
                    ),
                    "upstream_terminal_seen": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get("upstream_terminal_seen")
                        )
                    ),
                    "upstream_terminal_validated": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get(
                                "upstream_terminal_validated"
                            )
                        )
                    ),
                    "terminal_frame_seen": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get("terminal_frame_seen")
                        )
                    ),
                    "declared_terminal_type": _safe_text(
                        responses_diagnostics.get("declared_terminal_type")
                    ),
                    "declared_terminal_ordinal": _optional_int_text(
                        responses_diagnostics.get("declared_terminal_ordinal")
                    ),
                    "declared_terminal_bytes": _optional_int_text(
                        responses_diagnostics.get("declared_terminal_bytes")
                    ),
                    "declared_terminal_sha256": _safe_text(
                        responses_diagnostics.get("declared_terminal_sha256")
                    ),
                    "semantic_terminal_type": _safe_text(
                        responses_diagnostics.get("semantic_terminal_type")
                    ),
                    "semantic_terminal_outcome": _safe_text(
                        responses_diagnostics.get("semantic_terminal_outcome")
                    ),
                    "semantic_terminal_bytes": _optional_int_text(
                        responses_diagnostics.get("semantic_terminal_bytes")
                    ),
                    "semantic_terminal_sha256": _safe_text(
                        responses_diagnostics.get("semantic_terminal_sha256")
                    ),
                    "semantic_terminal_sequence_number": _optional_int_text(
                        responses_diagnostics.get(
                            "semantic_terminal_sequence_number"
                        )
                    ),
                    "downstream_terminal_seen": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get("downstream_terminal_seen")
                        )
                    ),
                    "ember_queue_terminal_handoff_completed": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get(
                                "ember_queue_terminal_handoff_completed"
                            )
                        )
                    ),
                    "downstream_terminal_asgi_write_completed": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get(
                                "downstream_terminal_asgi_write_completed"
                            )
                        )
                    ),
                    "error_event_seen": _bool_text(
                        _safe_bool(responses_diagnostics.get("error_event_seen"))
                    ),
                    "usage_seen": _bool_text(
                        _safe_bool(
                            responses_diagnostics.get(
                                "usage_seen",
                                current_info.get("usage_seen"),
                            )
                        )
                    ),
                    "diagnosis": _safe_text(
                        responses_diagnostics.get("diagnosis")
                    ),
                    "failure_stage": _responses_failure_stage(
                        current_info,
                        responses_diagnostics,
                    ),
                    "oaix_connection_id": _safe_text(
                        responses_diagnostics.get("oaix_connection_id")
                    ),
                    "upstream_body_bytes": _optional_int_text(
                        responses_diagnostics.get("upstream_body_bytes")
                    ),
                    "upstream_chunk_count": _optional_int_text(
                        responses_diagnostics.get("upstream_chunk_count")
                    ),
                    "last_event_type": _safe_text(
                        responses_diagnostics.get("last_event_type")
                    ),
                    "last_event_ordinal": _optional_int_text(
                        responses_diagnostics.get("last_event_ordinal")
                    ),
                    "last_event_bytes": _optional_int_text(
                        responses_diagnostics.get("last_event_bytes")
                    ),
                    "last_event_sha256": _safe_text(
                        responses_diagnostics.get("last_event_sha256")
                    ),
                    "partial_event_bytes": _optional_int_text(
                        responses_diagnostics.get("partial_event_bytes")
                    ),
                    "partial_event_sha256": _safe_text(
                        responses_diagnostics.get("partial_event_sha256")
                    ),
                    "event_hash_scope": _safe_text(
                        responses_diagnostics.get("hash_scope")
                    ),
                    "event_hash_policy": _safe_text(
                        responses_diagnostics.get("event_hash_policy")
                    ),
                    **_responses_diagnostic_attrs(
                        responses_diagnostics,
                        current_info=current_info,
                    ),
                }
            ),
        }
    ]
    logs.extend(_upstream_attempt_log_events(now, service_name, base, current_info))
    logs.extend(_routing_attempt_log_events(now, service_name, base, current_info))

    traces = []
    for stage, stage_ms, stage_attrs in _stage_rows(spans, duration_ms):
        traces.append(
            {
                "timestamp": _iso_timestamp(now),
                "kind": "span",
                "event_type": "request_span",
                "source": service_name,
                "message": stage,
                "attributes": _drop_empty(
                    {
                        **base,
                        **stage_attrs,
                        "span_id": _span_id(trace_id, request_id, stage),
                        "parent_span_id": _safe_text(current_info.get("parent_span_id") or spans.get("parent_span_id")),
                        "stage": stage,
                        "stage_ms": _int_text(stage_ms),
                    }
                ),
            }
        )
    if request_body_complexity_attrs:
        traces.append(
            {
                "timestamp": _iso_timestamp(now),
                "kind": "span",
                "event_type": "request_span",
                "source": service_name,
                "message": "request_body_rejected",
                "attributes": _drop_empty(
                    {
                        **base,
                        **request_body_complexity_attrs,
                        "span_id": _span_id(
                            trace_id,
                            request_id,
                            "request_body_rejected",
                        ),
                        "parent_span_id": _safe_text(
                            current_info.get("parent_span_id")
                            or spans.get("parent_span_id")
                        ),
                        "stage": "request_body_rejected",
                        "stage_ms": _int_text(
                            _span_ms(spans, "request_body_rejected")
                        ),
                    }
                ),
            }
        )

    metrics = _request_metric_events(
        service_name=service_name,
        identity_attrs=identity_attrs,
        timestamp=now,
        method=method,
        status_code=status_code,
        route_id=route_id,
        values={
            "uniapi_ember_request_duration_ms": duration_ms,
            "uniapi_ember_request_admission_wait_ms": _span_ms(
                spans, "request_admission_wait_ms"
            ),
            "uniapi_ember_request_ttfb_ms": ttft_ms,
            "uniapi_ember_inflight_requests": _runtime_int(runtime_metrics, "inflight_requests"),
            "uniapi_ember_request_waiters": _runtime_int(runtime_metrics, "request_waiters"),
            "uniapi_ember_cpu_phase_active": _runtime_int(
                runtime_metrics, "cpu_phase_active"
            ),
            "uniapi_ember_cpu_phase_waiters": _runtime_int(
                runtime_metrics, "cpu_phase_waiters"
            ),
            "uniapi_ember_request_large_body_active": _runtime_int(
                runtime_metrics, "request_large_body_active"
            ),
            "uniapi_ember_runtime_global_large_body_active": _runtime_int(
                runtime_metrics, "runtime_global_large_body_active"
            ),
            "uniapi_ember_request_body_reserved_weighted_bytes": _runtime_int(
                runtime_metrics, "request_body_reserved_weighted_bytes"
            ),
            "uniapi_ember_runtime_global_request_body_reserved_weighted_bytes": _runtime_int(
                runtime_metrics,
                "runtime_global_request_body_reserved_weighted_bytes",
            ),
            "uniapi_ember_upstream_response_reserved_weighted_bytes": _runtime_int(
                runtime_metrics, "upstream_response_reserved_weighted_bytes"
            ),
            "uniapi_ember_runtime_global_upstream_response_reserved_weighted_bytes": _runtime_int(
                runtime_metrics,
                "runtime_global_upstream_response_reserved_weighted_bytes",
            ),
            "uniapi_ember_request_retained_reserved_weighted_bytes": _runtime_int(
                runtime_metrics, "request_retained_reserved_weighted_bytes"
            ),
            "uniapi_ember_runtime_global_retained_reserved_weighted_bytes": _runtime_int(
                runtime_metrics,
                "runtime_global_retained_reserved_weighted_bytes",
            ),
            "uniapi_ember_request_deferred_memory_requests": _runtime_int(
                runtime_metrics, "request_deferred_memory_requests"
            ),
            "uniapi_ember_request_deferred_memory_weighted_bytes": _runtime_int(
                runtime_metrics, "request_deferred_memory_weighted_bytes"
            ),
            "uniapi_ember_runtime_global_deferred_memory_requests": _runtime_int(
                runtime_metrics, "runtime_global_deferred_memory_requests"
            ),
            "uniapi_ember_runtime_global_deferred_memory_weighted_bytes": _runtime_int(
                runtime_metrics, "runtime_global_deferred_memory_weighted_bytes"
            ),
            "uniapi_ember_waiting_first_byte": _runtime_int(runtime_metrics, "waiting_first_byte"),
            "uniapi_ember_event_loop_lag_ms": _runtime_int(runtime_metrics, "event_loop_lag_ms"),
            "uniapi_ember_client_pool_in_use": _runtime_int(runtime_metrics, "upstream_pool_in_use"),
            "uniapi_ember_client_pool_waiters": _runtime_int(
                runtime_metrics, "upstream_pool_waiters"
            ),
            "uniapi_ember_request_control_memory_reserved_bytes": _runtime_int(
                runtime_metrics,
                "request_control_memory_reserved_bytes",
            ),
            "uniapi_ember_outbound_fd_headroom": _safe_int(
                outbound_network_resources.get("fd_headroom"),
                0,
            ),
            "uniapi_ember_outbound_ephemeral_port_headroom": _safe_int(
                outbound_network_resources.get("ephemeral_port_headroom"),
                0,
            ),
            "uniapi_ember_client_pool_wait_ms": _span_ms(spans, "upstream_pool_wait_ms"),
            "uniapi_ember_stream_queue_bytes": _runtime_int(
                runtime_metrics, "stream_queue_bytes"
            ),
            "uniapi_ember_stream_queue_waiting_putters": _runtime_int(
                runtime_metrics, "stream_queue_waiting_putters"
            ),
            "uniapi_ember_stream_buffer_reserved_bytes": _runtime_int(
                runtime_metrics, "stream_buffer_reserved_bytes"
            ),
            "uniapi_ember_stream_buffer_budget_waiters": _runtime_int(
                runtime_metrics, "stream_buffer_budget_waiters"
            ),
            "uniapi_ember_stream_parser_reserved_bytes": _runtime_int(
                runtime_metrics, "stream_parser_reserved_bytes"
            ),
            "uniapi_ember_stream_parser_rejected_total": _runtime_int(
                runtime_metrics, "stream_parser_rejected_total"
            ),
            "uniapi_ember_stream_queue_peak_bytes": _safe_int(
                current_info.get("stream_queue_peak_bytes"), 0
            ),
            "uniapi_ember_retry_total": retry_count,
            "uniapi_ember_attempt_total": attempt_count,
            "uniapi_ember_retry_decision_total": retry_decision_count,
            "uniapi_ember_retry_transition_total": retry_transition_count,
            "uniapi_ember_provider_cooldown_total": cooldown_count,
            "uniapi_ember_upstream_errors_total": _actual_upstream_error_count(
                current_info
            ),
            "uniapi_ember_transport_errors_total": transport_error_count,
            "uniapi_ember_local_overload_total": local_overload_count,
            "uniapi_ember_exposed_5xx_total": 1 if status_code >= 500 else 0,
            "uniapi_ember_request_admission_rejected_total": 1
            if _safe_bool(current_info.get("admission_rejected"))
            else 0,
            "uniapi_ember_runtime_global_admission_rejection_decision_total": _runtime_int(
                runtime_metrics,
                "runtime_global_admission_rejection_decision_total",
            ),
            "uniapi_ember_runtime_global_admission_503_response_write_completed_total": _runtime_int(
                runtime_metrics,
                "runtime_global_admission_503_response_write_completed_total",
            ),
            "uniapi_ember_runtime_global_admission_503_response_write_failed_total": _runtime_int(
                runtime_metrics,
                "runtime_global_admission_503_response_write_failed_total",
            ),
            "uniapi_ember_runtime_global_large_body_decision_record_failures_total": _runtime_int(
                runtime_metrics,
                "runtime_global_large_body_decision_record_failures_total",
            ),
            "uniapi_ember_runtime_global_large_body_decision_observer_errors_total": _runtime_int(
                runtime_metrics,
                "runtime_global_large_body_decision_observer_errors_total",
            ),
            "uniapi_ember_runtime_global_large_body_decision_observer_enqueue_failures_total": _runtime_int(
                runtime_metrics,
                "runtime_global_large_body_decision_observer_enqueue_failures_total",
            ),
            "uniapi_ember_runtime_global_large_body_decision_export_enqueued_total": _runtime_int(
                runtime_metrics,
                "runtime_global_large_body_decision_export_enqueued_total",
            ),
            "uniapi_ember_runtime_global_large_body_decision_export_enqueue_dropped_total": _runtime_int(
                runtime_metrics,
                "runtime_global_large_body_decision_export_enqueue_dropped_total",
            ),
            "uniapi_ember_runtime_global_large_body_decision_export_build_errors_total": _runtime_int(
                runtime_metrics,
                "runtime_global_large_body_decision_export_build_errors_total",
            ),
            "uniapi_ember_runtime_global_large_body_decision_export_errors_total": _runtime_int(
                runtime_metrics,
                "runtime_global_large_body_decision_export_errors_total",
            ),
            "uniapi_ember_runtime_global_admission_503_outcome_export_enqueued_total": _runtime_int(
                runtime_metrics,
                "runtime_global_admission_503_outcome_export_enqueued_total",
            ),
            "uniapi_ember_runtime_global_admission_503_outcome_export_enqueue_dropped_total": _runtime_int(
                runtime_metrics,
                "runtime_global_admission_503_outcome_export_enqueue_dropped_total",
            ),
            "uniapi_ember_runtime_global_admission_503_outcome_export_build_errors_total": _runtime_int(
                runtime_metrics,
                "runtime_global_admission_503_outcome_export_build_errors_total",
            ),
            "uniapi_ember_runtime_global_admission_503_outcome_export_errors_total": _runtime_int(
                runtime_metrics,
                "runtime_global_admission_503_outcome_export_errors_total",
            ),
            "uniapi_ember_runtime_global_response_buffer_events_recorded_total": _runtime_int(
                runtime_metrics,
                "runtime_global_response_buffer_events_recorded_total",
            ),
            "uniapi_ember_runtime_global_response_buffer_event_record_failures_total": _runtime_int(
                runtime_metrics,
                "runtime_global_response_buffer_event_record_failures_total",
            ),
            "uniapi_ember_runtime_global_response_buffer_event_observer_errors_total": _runtime_int(
                runtime_metrics,
                "runtime_global_response_buffer_event_observer_errors_total",
            ),
            "uniapi_ember_runtime_global_response_buffer_event_observer_enqueue_failures_total": _runtime_int(
                runtime_metrics,
                "runtime_global_response_buffer_event_observer_enqueue_failures_total",
            ),
            "uniapi_ember_runtime_global_response_buffer_event_export_enqueued_total": _runtime_int(
                runtime_metrics,
                "runtime_global_response_buffer_event_export_enqueued_total",
            ),
            "uniapi_ember_runtime_global_response_buffer_event_export_enqueue_dropped_total": _runtime_int(
                runtime_metrics,
                "runtime_global_response_buffer_event_export_enqueue_dropped_total",
            ),
            "uniapi_ember_runtime_global_response_buffer_event_export_build_errors_total": _runtime_int(
                runtime_metrics,
                "runtime_global_response_buffer_event_export_build_errors_total",
            ),
            "uniapi_ember_runtime_global_response_buffer_event_export_errors_total": _runtime_int(
                runtime_metrics,
                "runtime_global_response_buffer_event_export_errors_total",
            ),
            "uniapi_ember_stream_failures_total": 1
            if _is_stream_failure(current_info)
            else 0,
            "uniapi_ember_downstream_disconnects_total": 1
            if _safe_bool(current_info.get("downstream_disconnected"))
            else 0,
        },
    )
    metrics.extend(
        _response_admission_metric_events(
            service_name=service_name,
            identity_attrs=identity_attrs,
            timestamp=now,
            runtime_metrics=runtime_metrics,
        )
    )
    return {"logs": logs, "traces": traces, "metrics": metrics}


def _request_body_complexity_attrs(
    current_info: dict[str, Any],
) -> dict[str, str]:
    """Allowlist body-free request diagnostics for Fugue ingestion."""

    raw = current_info.get("request_body_complexity")
    if not isinstance(raw, dict):
        return {}
    reason = _safe_text(raw.get("reason"), max_len=32)
    if reason not in _REQUEST_BODY_COMPLEXITY_REASONS:
        return {}
    trigger_phase = _safe_text(raw.get("trigger_phase"), max_len=32)
    if trigger_phase not in _REQUEST_BODY_COMPLEXITY_TRIGGER_PHASES:
        trigger_phase = None
    return _drop_empty(
        {
            "request_body_complexity_schema_version": _optional_int_text(
                raw.get("schema_version")
            ),
            "request_body_complexity_reason": reason,
            "request_body_complexity_trigger_phase": trigger_phase,
            "request_body_complexity_raw_bytes": _optional_int_text(
                raw.get("raw_bytes")
            ),
            "request_body_complexity_structural_item_count": (
                _optional_int_text(raw.get("structural_item_count"))
            ),
            "request_body_complexity_depth": _optional_int_text(
                raw.get("depth")
            ),
            "request_body_complexity_peak_depth": _optional_int_text(
                raw.get("peak_depth")
            ),
            "request_body_complexity_scalar_bytes": _optional_int_text(
                raw.get("scalar_bytes")
            ),
            "request_body_complexity_estimated_bytes": _optional_int_text(
                raw.get("estimated_bytes")
            ),
            "request_body_complexity_configured_limit": _optional_int_text(
                raw.get("configured_limit")
            ),
            "request_body_complexity_max_depth": _optional_int_text(
                raw.get("max_depth")
            ),
            "request_body_complexity_max_scalar_bytes": _optional_int_text(
                raw.get("max_scalar_bytes")
            ),
            "request_body_complexity_max_estimated_bytes": (
                _optional_int_text(raw.get("max_estimated_bytes"))
            ),
            "request_body_complexity_raw_memory_multiplier": (
                _optional_int_text(raw.get("raw_memory_multiplier"))
            ),
            "request_body_complexity_structural_item_memory_bytes": (
                _optional_int_text(raw.get("structural_item_memory_bytes"))
            ),
            "request_body_reserved_weighted_bytes_at_rejection": (
                _optional_int_text(
                    raw.get("reserved_weighted_bytes_at_rejection")
                )
            ),
            "json_memory_reserved_target_bytes_at_rejection": (
                _optional_int_text(
                    raw.get(
                        "json_memory_reserved_target_bytes_at_rejection"
                    )
                )
            ),
        }
    )


def _routing_attempt_log_events(
    timestamp: datetime,
    service_name: str,
    base: dict[str, str],
    current_info: dict[str, Any],
) -> list[dict[str, Any]]:
    attempts = current_info.get("routing_attempts")
    if not isinstance(attempts, list):
        return []

    events: list[dict[str, Any]] = []
    for raw_attempt in attempts[:32]:
        if not isinstance(raw_attempt, dict):
            continue
        semantic_status = _safe_int(
            raw_attempt.get("semantic_status_code"),
            0,
        )
        wire_status = _safe_int(raw_attempt.get("wire_status_code"), 0)
        effective_status = semantic_status or wire_status
        provider = _safe_text(raw_attempt.get("provider"))
        events.append(
            {
                "timestamp": _iso_timestamp(timestamp),
                "level": _event_level(effective_status),
                "service": service_name,
                "trace_id": base.get("trace_id"),
                "request_id": base.get("request_id"),
                "event": "routing_attempt",
                "event_type": "routing_attempt",
                "source": service_name,
                "message": "uni-api-ember routing attempt",
                "attributes": _drop_empty(
                    {
                        **base,
                        **_transport_phase_attrs(raw_attempt),
                        "provider": provider,
                        "channel": provider,
                        "model": _safe_text(raw_attempt.get("model"))
                        or base.get("model"),
                        "actual_model": _safe_text(
                            raw_attempt.get("actual_model")
                        ),
                        "attempt_index": _int_text(
                            _safe_int(raw_attempt.get("index"), 0)
                        ),
                        "attempt_outcome": _safe_text(
                            raw_attempt.get("outcome"),
                            max_len=80,
                        ),
                        "attempt_success": _bool_text(
                            _safe_bool(raw_attempt.get("success"))
                        )
                        if "success" in raw_attempt
                        else None,
                        "wire_status_code": _optional_int_text(
                            raw_attempt.get("wire_status_code")
                        ),
                        "semantic_status_code": _optional_int_text(
                            raw_attempt.get("semantic_status_code")
                        ),
                        "terminal_event_type": _safe_text(
                            raw_attempt.get("terminal_event_type"),
                            max_len=128,
                        ),
                        "attempt_error_code": _safe_text(
                            raw_attempt.get("error_code"),
                            max_len=128,
                        ),
                        "attempt_error_type": _safe_text(
                            raw_attempt.get("error_type"),
                            max_len=128,
                        ),
                        "transport_error_kind": _safe_text(
                            raw_attempt.get("transport_error_kind"),
                            max_len=80,
                        ),
                        "transport_error_owner": _safe_text(
                            raw_attempt.get("transport_error_owner"),
                            max_len=80,
                        ),
                        "transport_error_phase": _safe_text(
                            raw_attempt.get("transport_error_phase"),
                            max_len=80,
                        ),
                        "transport_error_status_code": _optional_int_text(
                            raw_attempt.get("transport_error_status_code")
                        ),
                        "provider_penalty_eligible": _bool_text(
                            _safe_bool(
                                raw_attempt.get(
                                    "provider_penalty_eligible"
                                )
                            )
                        )
                        if "provider_penalty_eligible" in raw_attempt
                        else None,
                        "local_overload": _bool_text(
                            _safe_bool(raw_attempt.get("local_overload"))
                        )
                        if "local_overload" in raw_attempt
                        else None,
                        "error_message_sha256": _safe_text(
                            raw_attempt.get("error_message_sha256"),
                            max_len=64,
                        ),
                        "error_message_hash_scope": _safe_text(
                            raw_attempt.get("error_message_hash_scope"),
                            max_len=80,
                        ),
                        "retry_decision": _bool_text(
                            _safe_bool(raw_attempt.get("retry_decision"))
                        )
                        if "retry_decision" in raw_attempt
                        else None,
                        "retry_reason": _safe_text(
                            raw_attempt.get("retry_reason"),
                            max_len=128,
                        ),
                        "retry_transition_to_index": _optional_int_text(
                            raw_attempt.get("retry_transition_to_index")
                        ),
                        "local_admission_rejected": _bool_text(
                            _safe_bool(
                                raw_attempt.get("local_admission_rejected")
                            )
                        )
                        if "local_admission_rejected" in raw_attempt
                        else None,
                        "status_origin": _safe_text(
                            raw_attempt.get("status_origin"),
                            max_len=64,
                        ),
                        "provider_model_circuit_opened": _bool_text(
                            _safe_bool(
                                raw_attempt.get(
                                    "provider_model_circuit_opened"
                                )
                            )
                        )
                        if "provider_model_circuit_opened" in raw_attempt
                        else None,
                        "provider_model_circuit_blocks_retry": _bool_text(
                            _safe_bool(
                                raw_attempt.get(
                                    "provider_model_circuit_blocks_retry"
                                )
                            )
                        )
                        if "provider_model_circuit_blocks_retry" in raw_attempt
                        else None,
                        "failure_stage": _safe_text(
                            raw_attempt.get("failure_stage"),
                            max_len=64,
                        ),
                        "protocol_error_reason": _safe_text(
                            raw_attempt.get("protocol_error_reason"),
                            max_len=128,
                        ),
                        "exception_type": _safe_text(
                            raw_attempt.get("exception_type"),
                            max_len=128,
                        ),
                        "exception_module": _safe_text(
                            raw_attempt.get("exception_module"),
                            max_len=256,
                        ),
                        "exception_repr": _safe_text(
                            raw_attempt.get("exception_repr"),
                            max_len=768,
                        ),
                        "exception_chain_json": _safe_text(
                            raw_attempt.get("exception_chain_json"),
                            max_len=4096,
                        ),
                        "httpcore_exception_type": _safe_text(
                            raw_attempt.get("httpcore_exception_type"),
                            max_len=128,
                        ),
                        "httpcore_exception_module": _safe_text(
                            raw_attempt.get("httpcore_exception_module"),
                            max_len=256,
                        ),
                        "httpcore_exception_repr": _safe_text(
                            raw_attempt.get("httpcore_exception_repr"),
                            max_len=768,
                        ),
                        "httpcore_exception_chain_json": _safe_text(
                            raw_attempt.get("httpcore_exception_chain_json"),
                            max_len=4096,
                        ),
                        "http_version": _safe_text(
                            raw_attempt.get("http_version"),
                            max_len=32,
                        ),
                        "upstream_http_status_code": _optional_int_text(
                            raw_attempt.get("upstream_http_status_code")
                        ),
                        "alpn_protocol": _safe_text(
                            raw_attempt.get("alpn_protocol"),
                            max_len=32,
                        ),
                        "http2_stream_id": _optional_int_text(
                            raw_attempt.get("http2_stream_id")
                        ),
                        "connection_request_count": _optional_int_text(
                            raw_attempt.get("connection_request_count")
                        ),
                        "http2_concurrent_streams": _optional_int_text(
                            raw_attempt.get("http2_concurrent_streams")
                        ),
                        "http2_max_concurrent_streams": _optional_int_text(
                            raw_attempt.get("http2_max_concurrent_streams")
                        ),
                        "connection_local_state": _safe_text(
                            raw_attempt.get("connection_local_state"),
                            max_len=64,
                        ),
                        "http2_local_connection_state": _safe_text(
                            raw_attempt.get("http2_local_connection_state"),
                            max_len=64,
                        ),
                        "http2_local_stream_state": _safe_text(
                            raw_attempt.get("http2_local_stream_state"),
                            max_len=64,
                        ),
                        "goaway_error_code": _optional_int_text(
                            raw_attempt.get("goaway_error_code")
                        ),
                        "goaway_error_code_name": _safe_text(
                            raw_attempt.get("goaway_error_code_name"),
                            max_len=128,
                        ),
                        "goaway_last_stream_id": _optional_int_text(
                            raw_attempt.get("goaway_last_stream_id")
                        ),
                        "connection_snapshot_json": _safe_text(
                            raw_attempt.get("connection_snapshot_json"),
                            max_len=4096,
                        ),
                        "httpcore_events_json": _safe_text(
                            raw_attempt.get("httpcore_events_json"),
                            max_len=8192,
                        ),
                        "response_buffer_reserved_before_bytes": _optional_int_text(
                            raw_attempt.get(
                                "response_buffer_reserved_before_bytes"
                            )
                        ),
                        "response_buffer_reserved_after_bytes": _optional_int_text(
                            raw_attempt.get(
                                "response_buffer_reserved_after_bytes"
                            )
                        ),
                        "response_buffer_retained_from_prior_attempts_bytes": _optional_int_text(
                            raw_attempt.get(
                                "response_buffer_retained_from_prior_attempts_bytes"
                            )
                        ),
                        "response_buffer_retained_after_failed_attempt": _bool_text(
                            _safe_bool(
                                raw_attempt.get(
                                    "response_buffer_retained_after_failed_attempt"
                                )
                            )
                        )
                        if "response_buffer_retained_after_failed_attempt"
                        in raw_attempt
                        else None,
                        "started_ms": _optional_int_text(
                            raw_attempt.get("started_ms")
                        ),
                        "duration_ms": _optional_int_text(
                            raw_attempt.get("duration_ms")
                        ),
                    }
                ),
            }
        )
    return events


def _upstream_attempt_log_events(
    timestamp: datetime,
    service_name: str,
    base: dict[str, str],
    current_info: dict[str, Any],
) -> list[dict[str, Any]]:
    attempts = current_info.get("upstream_attempts")
    if not isinstance(attempts, list):
        return []

    events: list[dict[str, Any]] = []
    for raw_attempt in attempts[:16]:
        if not isinstance(raw_attempt, dict):
            continue
        attempt_status = _safe_int(raw_attempt.get("status_code"), 0)
        attempt_provider = _safe_text(raw_attempt.get("provider"))
        attempt_error_type = _safe_text(raw_attempt.get("error_type"), max_len=80)
        stream_diagnostics = raw_attempt.get("stream_diagnostics")
        if not isinstance(stream_diagnostics, dict):
            stream_diagnostics = {}
        timeout_adjusted_from = _optional_int_text(raw_attempt.get("timeout_adjusted_from_seconds"))
        started_ms = _optional_int_text(raw_attempt.get("started_ms"))
        duration_ms = _optional_int_text(raw_attempt.get("duration_ms"))
        events.append(
            {
                "timestamp": _iso_timestamp(timestamp),
                "level": _event_level(attempt_status),
                "service": service_name,
                "trace_id": base.get("trace_id"),
                "request_id": base.get("request_id"),
                "event": "upstream_attempt",
                "event_type": "upstream_attempt",
                "source": service_name,
                "message": "uni-api-ember upstream attempt",
                "attributes": _drop_empty(
                    {
                        **base,
                        **_transport_phase_attrs(
                            stream_diagnostics,
                            fallback=raw_attempt,
                        ),
                        "provider": attempt_provider,
                        "channel": attempt_provider,
                        "model": _safe_text(raw_attempt.get("model")) or base.get("model"),
                        "actual_model": _safe_text(raw_attempt.get("actual_model")),
                        "engine": _safe_text(raw_attempt.get("engine")),
                        "upstream_host": _safe_text(raw_attempt.get("upstream_host")),
                        "attempt_index": _int_text(_safe_int(raw_attempt.get("index"), 0)),
                        "attempt_status_code": _int_text(attempt_status),
                        "attempt_status_class": _status_class(attempt_status),
                        "attempt_success": _bool_text(_safe_bool(raw_attempt.get("success"))),
                        "attempt_error_type": attempt_error_type,
                        "transport_error_kind": _safe_text(
                            stream_diagnostics.get("transport_error_kind")
                            or raw_attempt.get("transport_error_kind"),
                            max_len=80,
                        ),
                        "transport_error_owner": _safe_text(
                            stream_diagnostics.get("transport_error_owner")
                            or raw_attempt.get("transport_error_owner"),
                            max_len=80,
                        ),
                        "transport_error_phase": _safe_text(
                            stream_diagnostics.get("transport_error_phase")
                            or raw_attempt.get("transport_error_phase"),
                            max_len=80,
                        ),
                        "transport_error_status_code": _optional_int_text(
                            stream_diagnostics.get(
                                "transport_error_status_code"
                            )
                            or raw_attempt.get(
                                "transport_error_status_code"
                            )
                        ),
                        "provider_penalty_eligible": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "provider_penalty_eligible"
                                )
                                if "provider_penalty_eligible"
                                in stream_diagnostics
                                else raw_attempt.get(
                                    "provider_penalty_eligible"
                                )
                            )
                        )
                        if (
                            "provider_penalty_eligible"
                            in stream_diagnostics
                            or "provider_penalty_eligible" in raw_attempt
                        )
                        else None,
                        "local_overload": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get("local_overload")
                                if "local_overload" in stream_diagnostics
                                else raw_attempt.get("local_overload")
                            )
                        )
                        if (
                            "local_overload" in stream_diagnostics
                            or "local_overload" in raw_attempt
                        )
                        else None,
                        "status_origin": _safe_text(
                            raw_attempt.get("status_origin"),
                            max_len=64,
                        ),
                        "payload_bytes": _int_text(_safe_int(raw_attempt.get("payload_bytes"), 0)),
                        "timeout_seconds": _int_text(_safe_int(raw_attempt.get("timeout_seconds"), 0)),
                        "timeout_adjusted_from_seconds": timeout_adjusted_from,
                        "wants_compact": _bool_text(_safe_bool(raw_attempt.get("wants_compact"))),
                        "stream": _bool_text(_safe_bool(raw_attempt.get("stream"))),
                        "started_ms": started_ms,
                        "duration_ms": duration_ms,
                        "semantic_status": _safe_text(
                            stream_diagnostics.get("semantic_status")
                        ),
                        "diagnosis": _safe_text(
                            stream_diagnostics.get("diagnosis")
                        ),
                        "failure_stage": _safe_text(
                            stream_diagnostics.get("failure_stage")
                            or raw_attempt.get("failure_stage")
                        ),
                        "oaix_connection_id": _safe_text(
                            stream_diagnostics.get("oaix_connection_id")
                        ),
                        "upstream_http_version": _safe_text(
                            stream_diagnostics.get("http_version")
                            or raw_attempt.get("http_version")
                        ),
                        "httpcore_stream_id": _optional_int_text(
                            stream_diagnostics.get("httpcore_stream_id")
                            or raw_attempt.get("http2_stream_id")
                        ),
                        "protocol_error_reason": _safe_text(
                            raw_attempt.get("protocol_error_reason"),
                            max_len=128,
                        ),
                        "exception_module": _safe_text(
                            raw_attempt.get("exception_module"),
                            max_len=256,
                        ),
                        "exception_repr": _safe_text(
                            raw_attempt.get("exception_repr"),
                            max_len=768,
                        ),
                        "exception_chain_json": _safe_text(
                            raw_attempt.get("exception_chain_json"),
                            max_len=4096,
                        ),
                        "alpn_protocol": _safe_text(
                            raw_attempt.get("alpn_protocol"),
                            max_len=32,
                        ),
                        "connection_request_count": _optional_int_text(
                            raw_attempt.get("connection_request_count")
                        ),
                        "http2_concurrent_streams": _optional_int_text(
                            raw_attempt.get("http2_concurrent_streams")
                        ),
                        "http2_max_concurrent_streams": _optional_int_text(
                            raw_attempt.get("http2_max_concurrent_streams")
                        ),
                        "http2_local_connection_state": _safe_text(
                            raw_attempt.get("http2_local_connection_state"),
                            max_len=64,
                        ),
                        "http2_local_stream_state": _safe_text(
                            raw_attempt.get("http2_local_stream_state"),
                            max_len=64,
                        ),
                        "goaway_error_code": _optional_int_text(
                            raw_attempt.get("goaway_error_code")
                        ),
                        "explicit_proxy_configured": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "explicit_proxy_configured"
                                )
                            )
                        ),
                        "transport_local_endpoint_hmac": _safe_text(
                            stream_diagnostics.get(
                                "transport_local_endpoint_hmac"
                            )
                        ),
                        "transport_peer_endpoint_hmac": _safe_text(
                            stream_diagnostics.get(
                                "transport_peer_endpoint_hmac"
                            )
                        ),
                        "transport_four_tuple_hmac": _safe_text(
                            stream_diagnostics.get("transport_four_tuple_hmac")
                        ),
                        "transport_socket_hmac": _safe_text(
                            stream_diagnostics.get("transport_socket_hmac")
                        ),
                        "transport_local_family": _safe_text(
                            stream_diagnostics.get("transport_local_family")
                        ),
                        "transport_peer_family": _safe_text(
                            stream_diagnostics.get("transport_peer_family")
                        ),
                        "upstream_body_bytes": _optional_int_text(
                            stream_diagnostics.get("upstream_body_bytes")
                        ),
                        "upstream_chunk_count": _optional_int_text(
                            stream_diagnostics.get("upstream_chunk_count")
                        ),
                        "complete_event_count": _optional_int_text(
                            stream_diagnostics.get("complete_event_count")
                        ),
                        "last_event_type": _safe_text(
                            stream_diagnostics.get("last_event_type")
                        ),
                        "last_event_ordinal": _optional_int_text(
                            stream_diagnostics.get("last_event_ordinal")
                        ),
                        "last_event_bytes": _optional_int_text(
                            stream_diagnostics.get("last_event_bytes")
                        ),
                        "last_event_sha256": _safe_text(
                            stream_diagnostics.get("last_event_sha256")
                        ),
                        "partial_event_bytes": _optional_int_text(
                            stream_diagnostics.get("partial_event_bytes")
                        ),
                        "partial_event_sha256": _safe_text(
                            stream_diagnostics.get("partial_event_sha256")
                        ),
                        "event_hash_scope": _safe_text(
                            stream_diagnostics.get("hash_scope")
                        ),
                        "event_hash_policy": _safe_text(
                            stream_diagnostics.get("event_hash_policy")
                        ),
                        "partial_hash_scope": _safe_text(
                            stream_diagnostics.get("partial_hash_scope")
                        ),
                        "upstream_eof_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get("upstream_eof_seen")
                            )
                        ),
                        "upstream_terminal_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "upstream_terminal_seen"
                                )
                            )
                        ),
                        "upstream_terminal_validated": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "upstream_terminal_validated"
                                )
                            )
                        ),
                        "terminal_frame_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get("terminal_frame_seen")
                            )
                        ),
                        "declared_terminal_type": _safe_text(
                            stream_diagnostics.get("declared_terminal_type")
                        ),
                        "declared_terminal_ordinal": _optional_int_text(
                            stream_diagnostics.get("declared_terminal_ordinal")
                        ),
                        "declared_terminal_bytes": _optional_int_text(
                            stream_diagnostics.get("declared_terminal_bytes")
                        ),
                        "declared_terminal_sha256": _safe_text(
                            stream_diagnostics.get("declared_terminal_sha256")
                        ),
                        "semantic_terminal_type": _safe_text(
                            stream_diagnostics.get("semantic_terminal_type")
                        ),
                        "semantic_terminal_outcome": _safe_text(
                            stream_diagnostics.get("semantic_terminal_outcome")
                        ),
                        "semantic_terminal_bytes": _optional_int_text(
                            stream_diagnostics.get("semantic_terminal_bytes")
                        ),
                        "semantic_terminal_sha256": _safe_text(
                            stream_diagnostics.get("semantic_terminal_sha256")
                        ),
                        "semantic_terminal_sequence_number": _optional_int_text(
                            stream_diagnostics.get(
                                "semantic_terminal_sequence_number"
                            )
                        ),
                        "downstream_terminal_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "downstream_terminal_seen"
                                )
                            )
                        ),
                        "ember_queue_terminal_handoff_completed": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "ember_queue_terminal_handoff_completed"
                                )
                            )
                        ),
                        "downstream_terminal_asgi_write_completed": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "downstream_terminal_asgi_write_completed"
                                )
                            )
                        ),
                        "error_event_seen": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get("error_event_seen")
                            )
                        ),
                        "usage_seen": _bool_text(
                            _safe_bool(stream_diagnostics.get("usage_seen"))
                        ),
                        "exception_type": _safe_text(
                            stream_diagnostics.get("exception_type"),
                            max_len=80,
                        ),
                        "exception_origin": _safe_text(
                            stream_diagnostics.get("exception_origin"),
                            max_len=80,
                        ),
                        "exception_errno": _optional_int_text(
                            stream_diagnostics.get("exception_errno")
                        ),
                        "exception_errno_name": _safe_text(
                            stream_diagnostics.get("exception_errno_name"),
                            max_len=80,
                        ),
                        "exception_chain_depth": _optional_int_text(
                            stream_diagnostics.get("exception_chain_depth")
                        ),
                        "exception_chain_truncated": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "exception_chain_truncated"
                                )
                            )
                        ),
                        "exception_chain_json": _diagnostic_json(
                            stream_diagnostics.get("exception_chain")
                        ),
                        "httpcore_events_json": _diagnostic_json(
                            stream_diagnostics.get("httpcore_events")
                        ),
                        "httpcore_response_close_trigger": _safe_text(
                            stream_diagnostics.get(
                                "httpcore_response_close_trigger"
                            )
                        ),
                        "cleanup_owner": _safe_text(
                            stream_diagnostics.get("cleanup_owner")
                        ),
                        "cleanup_trigger": _safe_text(
                            stream_diagnostics.get("cleanup_trigger")
                        ),
                        "cleanup_method": _safe_text(
                            stream_diagnostics.get("cleanup_method")
                        ),
                        "cleanup_result": _safe_text(
                            stream_diagnostics.get("cleanup_result")
                        ),
                        "cleanup_transport_evicted": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "cleanup_transport_evicted"
                                )
                            )
                        ),
                        "cleanup_transport_safe": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get("cleanup_transport_safe")
                            )
                        ),
                        "cleanup_detached": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "cleanup_detached_cleanup"
                                )
                            )
                        ),
                        "pool_sweeper_close_observed": _bool_text(
                            _safe_bool(
                                stream_diagnostics.get(
                                    "pool_sweeper_close_observed"
                                )
                            )
                        ),
                        "pool_sweeper_trigger": _safe_text(
                            stream_diagnostics.get("pool_sweeper_trigger")
                        ),
                        **_responses_diagnostic_attrs(
                            stream_diagnostics,
                            current_info=current_info,
                        ),
                    }
                ),
            }
        )
    return events


def _stage_rows(spans: dict[str, Any], duration_ms: int | None) -> list[tuple[str, int, dict[str, str]]]:
    rows: list[tuple[str, int, dict[str, str]]] = []
    previous_stage = ""
    for stage in _STAGE_ORDER:
        observed = stage in spans
        if not observed:
            continue
        if stage == "client_pool_acquired":
            stage_ms = _span_ms(spans, "upstream_pool_wait_ms")
            attrs = {
                "client_pool_acquire_start_ms": _int_text(_span_ms(spans, "client_pool_acquire_start")),
                "client_pool_acquire_end_ms": _int_text(_span_ms(spans, "client_pool_acquire_end")),
            }
        elif stage == "retry_started":
            stage_ms = _stage_delta_ms(spans, stage, previous_stage)
            attrs = {
                "retry_count": _int_text(_span_ms(spans, "retry_count")),
                "retry_status_code": _int_text(_span_ms(spans, "retry_status_code")),
                "retry_provider": _safe_text(spans.get("retry_provider")),
            }
        elif stage == "stream_end" and _span_ms(spans, stage) <= 0 and duration_ms is not None:
            stage_ms = max(0, int(duration_ms))
            attrs = {}
        elif stage == "upstream_first_chunk":
            stage_ms = _span_ms(spans, stage)
            attrs = {}
        else:
            stage_ms = _stage_delta_ms(spans, stage, previous_stage)
            attrs = {}
        rows.append((stage, max(0, int(stage_ms or 0)), attrs))
        if _span_ms(spans, stage) > 0 or stage == "request_received":
            previous_stage = stage
    return rows


def _request_metric_events(
    *,
    service_name: str,
    identity_attrs: dict[str, str] | None,
    timestamp: datetime,
    method: str | None,
    status_code: int,
    route_id: str | None,
    values: dict[str, int | None],
) -> list[dict[str, Any]]:
    request_attrs = _drop_empty(
        {
            **(identity_attrs or {}),
            "component": service_name,
            "metric_scope": "request_self",
            "route_id": route_id,
            "method": method,
            "status_class": _status_class(status_code),
        }
    )
    global_attrs = _drop_empty(
        {
            **(identity_attrs or {}),
            "component": service_name,
            "metric_scope": "runtime_global",
        }
    )
    legacy_global_aliases = {
        "uniapi_ember_request_large_body_active": (
            "uniapi_ember_runtime_global_large_body_active"
        ),
        "uniapi_ember_request_body_reserved_weighted_bytes": (
            "uniapi_ember_runtime_global_request_body_reserved_weighted_bytes"
        ),
        "uniapi_ember_upstream_response_reserved_weighted_bytes": (
            "uniapi_ember_runtime_global_upstream_response_reserved_weighted_bytes"
        ),
        "uniapi_ember_request_retained_reserved_weighted_bytes": (
            "uniapi_ember_runtime_global_retained_reserved_weighted_bytes"
        ),
        "uniapi_ember_request_deferred_memory_requests": (
            "uniapi_ember_runtime_global_deferred_memory_requests"
        ),
        "uniapi_ember_request_deferred_memory_weighted_bytes": (
            "uniapi_ember_runtime_global_deferred_memory_weighted_bytes"
        ),
    }
    global_metrics = {
        "uniapi_ember_inflight_requests",
        "uniapi_ember_request_waiters",
        "uniapi_ember_cpu_phase_active",
        "uniapi_ember_cpu_phase_waiters",
        "uniapi_ember_request_control_memory_reserved_bytes",
        "uniapi_ember_outbound_fd_headroom",
        "uniapi_ember_outbound_ephemeral_port_headroom",
        "uniapi_ember_request_large_body_active",
        "uniapi_ember_runtime_global_large_body_active",
        "uniapi_ember_request_body_reserved_weighted_bytes",
        "uniapi_ember_runtime_global_request_body_reserved_weighted_bytes",
        "uniapi_ember_upstream_response_reserved_weighted_bytes",
        "uniapi_ember_runtime_global_upstream_response_reserved_weighted_bytes",
        "uniapi_ember_request_retained_reserved_weighted_bytes",
        "uniapi_ember_runtime_global_retained_reserved_weighted_bytes",
        "uniapi_ember_request_deferred_memory_requests",
        "uniapi_ember_request_deferred_memory_weighted_bytes",
        "uniapi_ember_runtime_global_deferred_memory_requests",
        "uniapi_ember_runtime_global_deferred_memory_weighted_bytes",
        "uniapi_ember_waiting_first_byte",
        "uniapi_ember_event_loop_lag_ms",
        "uniapi_ember_client_pool_in_use",
        "uniapi_ember_client_pool_waiters",
        "uniapi_ember_stream_queue_bytes",
        "uniapi_ember_stream_queue_waiting_putters",
        "uniapi_ember_stream_buffer_reserved_bytes",
        "uniapi_ember_stream_buffer_budget_waiters",
        "uniapi_ember_stream_parser_reserved_bytes",
        "uniapi_ember_runtime_global_admission_rejection_decision_total",
        "uniapi_ember_runtime_global_admission_503_response_write_completed_total",
        "uniapi_ember_runtime_global_admission_503_response_write_failed_total",
        "uniapi_ember_runtime_global_large_body_decision_record_failures_total",
        "uniapi_ember_runtime_global_large_body_decision_observer_errors_total",
        "uniapi_ember_runtime_global_large_body_decision_observer_enqueue_failures_total",
        "uniapi_ember_runtime_global_large_body_decision_export_enqueued_total",
        "uniapi_ember_runtime_global_large_body_decision_export_enqueue_dropped_total",
        "uniapi_ember_runtime_global_large_body_decision_export_build_errors_total",
        "uniapi_ember_runtime_global_large_body_decision_export_errors_total",
        "uniapi_ember_runtime_global_admission_503_outcome_export_enqueued_total",
        "uniapi_ember_runtime_global_admission_503_outcome_export_enqueue_dropped_total",
        "uniapi_ember_runtime_global_admission_503_outcome_export_build_errors_total",
        "uniapi_ember_runtime_global_admission_503_outcome_export_errors_total",
        "uniapi_ember_runtime_global_response_buffer_events_recorded_total",
        "uniapi_ember_runtime_global_response_buffer_event_record_failures_total",
        "uniapi_ember_runtime_global_response_buffer_event_observer_errors_total",
        "uniapi_ember_runtime_global_response_buffer_event_observer_enqueue_failures_total",
        "uniapi_ember_runtime_global_response_buffer_event_export_enqueued_total",
        "uniapi_ember_runtime_global_response_buffer_event_export_enqueue_dropped_total",
        "uniapi_ember_runtime_global_response_buffer_event_export_build_errors_total",
        "uniapi_ember_runtime_global_response_buffer_event_export_errors_total",
    }
    events = []
    for metric, value in values.items():
        if value is None:
            continue
        attributes = dict(global_attrs if metric in global_metrics else request_attrs)
        legacy_alias_of = legacy_global_aliases.get(metric)
        if legacy_alias_of is not None:
            attributes["legacy_alias_of"] = legacy_alias_of
        events.append(
            {
                "timestamp": _iso_timestamp(timestamp),
                "kind": "metric",
                "source": service_name,
                "message": metric,
                "metric": metric,
                "value": max(0, int(value)),
                "attributes": attributes,
            }
        )
    return events


def _response_admission_metric_events(
    *,
    service_name: str,
    identity_attrs: dict[str, str] | None,
    timestamp: datetime,
    runtime_metrics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    runtime = runtime_metrics if isinstance(runtime_metrics, dict) else {}
    branches = runtime.get(
        "runtime_global_response_admission_rejections_by_branch"
    )
    if not isinstance(branches, dict):
        branches = {}
    allowed_branches = (
        "parent_governor",
        "per_request_response_limit",
        "per_request_retained_limit",
        "global_hard_budget",
    )
    base = _drop_empty(
        {
            **(identity_attrs or {}),
            "component": service_name,
            "metric_scope": "runtime_global",
        }
    )
    rows: list[tuple[str, int, dict[str, Any]]] = []
    for branch in allowed_branches:
        rows.append(
            (
                "uniapi_ember_runtime_global_response_admission_rejections_total",
                max(0, _safe_int(branches.get(branch), 0)),
                {"admission_branch": branch},
            )
        )
    remaining = _runtime_int(
        runtime,
        "runtime_global_response_budget_soft_remaining_bytes",
    )
    if remaining is not None:
        rows.append(
            (
                "uniapi_ember_runtime_global_response_budget_soft_remaining_bytes",
                max(0, remaining),
                {},
            )
        )
    ratio = runtime.get("runtime_global_response_budget_soft_remaining_ratio")
    try:
        ratio_basis_points = max(0, min(10000, int(float(ratio) * 10000)))
    except (TypeError, ValueError):
        ratio_basis_points = None
    if ratio_basis_points is not None:
        rows.append(
            (
                "uniapi_ember_runtime_global_response_budget_soft_remaining_ratio_basis_points",
                ratio_basis_points,
                {},
            )
        )
    rejections_1m = _runtime_int(
        runtime,
        "runtime_global_response_admission_rejections_1m",
    )
    if rejections_1m is not None:
        rows.append(
            (
                "uniapi_ember_runtime_global_response_admission_rejections_1m",
                max(0, rejections_1m),
                {},
            )
        )
    for alert_name, runtime_key in (
        (
            "uniapi_ember_runtime_global_response_budget_soft_headroom_alert",
            "runtime_global_response_budget_soft_headroom_alert",
        ),
        (
            "uniapi_ember_runtime_global_response_rejection_rate_alert",
            "runtime_global_response_rejection_rate_alert",
        ),
    ):
        if runtime_key in runtime:
            rows.append((alert_name, 1 if _safe_bool(runtime[runtime_key]) else 0, {}))
    return [
        {
            "timestamp": _iso_timestamp(timestamp),
            "kind": "metric",
            "source": service_name,
            "message": metric,
            "metric": metric,
            "value": value,
            "attributes": _drop_empty({**base, **attrs}),
        }
        for metric, value, attrs in rows
    ]


def _actual_upstream_error_count(current_info: dict[str, Any]) -> int | None:
    attempts = current_info.get("upstream_attempts")
    if not isinstance(attempts, list):
        # Legacy routes do not yet expose unified attempt facts.  Omitting the
        # metric is truthful; emitting zero would silently claim knowledge we
        # do not have.
        return None
    count = 0
    for attempt in attempts[:16]:
        if not isinstance(attempt, dict) or _safe_bool(attempt.get("success")):
            continue
        if _safe_bool(attempt.get("local_admission_rejected")):
            continue
        status_code = _safe_int(attempt.get("status_code"), 0)
        error_type = _safe_text(attempt.get("error_type")) or ""
        if error_type in {"UpstreamAdmissionRejected", "StreamBufferBudgetTimeout"}:
            continue
        if status_code >= 500:
            count += 1
    return count


def _responses_stream_diagnostics(current_info: dict[str, Any]) -> dict[str, Any]:
    diagnostics = current_info.get("responses_stream_diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _is_responses_request(current_info: dict[str, Any]) -> bool:
    """Responses correlation logs are never sampled away.

    A request may look completely healthy inside Ember while a downstream
    consumer fails to parse its terminal usage.  Retaining every compact
    Responses summary is what makes a later 0-0 ``200 / unknown usage`` row
    joinable by request/connection/event hashes.
    """

    _method, path = _split_endpoint(_safe_text(current_info.get("endpoint")))
    if not path:
        return False
    normalized = path.split("?", 1)[0].rstrip("/")
    return normalized in {"/v1/responses", "/v1/responses/compact"}


def _responses_token_text(
    current_info: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    value_key: str,
    known_key: str,
) -> str | None:
    if _is_responses_request(current_info) and not diagnostics:
        if _safe_bool(current_info.get("usage_seen")) is not True:
            return None
    if diagnostics:
        if _safe_bool(diagnostics.get(known_key)) is not True:
            return None
        if _safe_bool(diagnostics.get("downstream_usage_values_valid")) is not True:
            return None
        if _safe_bool(diagnostics.get("downstream_usage_alias_consistent")) is False:
            return None
    return _optional_int_text(current_info.get(value_key))


def _responses_failure_stage(
    current_info: dict[str, Any],
    diagnostics: dict[str, Any],
) -> str | None:
    if not diagnostics:
        return None
    explicit_stage = (_safe_text(diagnostics.get("failure_stage")) or "").lower()
    if explicit_stage in {"precommit", "postcommit", "downstream", "cleanup"}:
        return explicit_stage
    if explicit_stage in {
        "upstream",
        "upstream_headers",
        "upstream_response_headers",
        "headers",
    } or explicit_stage.startswith("precommit"):
        return "precommit"
    if explicit_stage.startswith("postcommit"):
        return "postcommit"
    if explicit_stage.startswith("downstream"):
        return "downstream"
    if explicit_stage.startswith("cleanup"):
        return "cleanup"
    if diagnostics.get("cleanup_result") == "incomplete":
        return "cleanup"
    outcome = _safe_text(current_info.get("stream_outcome")) or ""
    if outcome in {
        "downstream_disconnected",
        "downstream_write_timeout",
        "downstream_send_error",
        "local_backpressure_abort",
    }:
        return "downstream"
    origin = _safe_text(diagnostics.get("exception_origin")) or ""
    if origin.startswith("precommit") or origin == "upstream_response_headers":
        return "precommit"
    if diagnostics.get("exception_type") or diagnostics.get(
        "failure_terminal_seen"
    ):
        return "postcommit" if current_info.get("response_committed") else "precommit"
    return None


def _diagnostic_json(value: Any, *, max_len: int = 4096) -> str | None:
    encoded, _truncated, _count, _digest = _diagnostic_json_details(
        value,
        max_len=max_len,
    )
    return encoded


def _diagnostic_json_details(
    value: Any,
    *,
    max_len: int = 4096,
) -> tuple[str | None, bool, int | None, str | None]:
    if value in (None, [], {}):
        return None, False, None, None
    try:
        full = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return None, False, None, None

    full_bytes = full.encode("utf-8")
    digest = hashlib.sha256(full_bytes).hexdigest()
    count = len(value) if isinstance(value, (list, dict)) else 1
    if len(full_bytes) <= max_len:
        return full, False, count, digest

    if isinstance(value, list):
        bounded: Any = []
        for item in value:
            candidate = [*bounded, item]
            try:
                rendered = json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                break
            if len(rendered.encode("utf-8")) > max_len:
                break
            bounded = candidate
    elif isinstance(value, dict):
        bounded = {}
        for key in sorted(value, key=lambda candidate: str(candidate)):
            candidate = {**bounded, key: value[key]}
            try:
                rendered = json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                break
            if len(rendered.encode("utf-8")) > max_len:
                break
            bounded = candidate
    else:
        bounded = {"truncated": True, "value_type": type(value).__name__}

    encoded = json.dumps(
        bounded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > max_len:
        encoded = "[]" if isinstance(value, list) else "{}"
    return encoded, True, count, digest


def _diagnostic_json_attrs(
    prefix: str,
    value: Any,
    *,
    source_truncated: bool = False,
) -> dict[str, str | None]:
    encoded, bounded_truncated, count, digest = _diagnostic_json_details(value)
    if encoded is None:
        return {}
    return {
        f"{prefix}_json": encoded,
        f"{prefix}_json_truncated": _bool_text(
            bool(source_truncated or bounded_truncated)
        ),
        f"{prefix}_original_count": _optional_int_text(count),
        f"{prefix}_json_sha256": digest,
    }


def _responses_diagnostic_attrs(
    diagnostics: dict[str, Any],
    *,
    current_info: dict[str, Any],
) -> dict[str, str]:
    if not diagnostics:
        return {}

    attrs: dict[str, Any] = {
        "responses_diagnostic_schema_version": _optional_int_text(
            diagnostics.get("schema_version")
        ),
        "semantic_status": _safe_text(diagnostics.get("semantic_status")),
        "diagnosis": _safe_text(diagnostics.get("diagnosis")),
        "failure_stage": _responses_failure_stage(current_info, diagnostics),
        "terminal_consistency_status": _safe_text(
            diagnostics.get("terminal_consistency_status")
        ),
        "declared_terminal_type": _safe_text(
            diagnostics.get("declared_terminal_type")
        ),
        "semantic_terminal_type": _safe_text(
            diagnostics.get("semantic_terminal_type")
        ),
        "semantic_terminal_outcome": _safe_text(
            diagnostics.get("semantic_terminal_outcome")
        ),
        "downstream_declared_terminal_type": _safe_text(
            diagnostics.get("downstream_declared_terminal_type")
        ),
        "downstream_semantic_status": _safe_text(
            diagnostics.get("downstream_semantic_status")
        ),
        "oaix_connection_id": _safe_text(
            diagnostics.get("oaix_connection_id")
        ),
        "upstream_http_version": _safe_text(diagnostics.get("http_version")),
        "transport_error_code": _safe_text(
            diagnostics.get("transport_error_code")
        ),
        "transport_error_code_source": _safe_text(
            diagnostics.get("transport_error_code_source")
        ),
        "transport_error_kind": _safe_text(
            diagnostics.get("transport_error_kind"), max_len=80
        ),
        "transport_error_owner": _safe_text(
            diagnostics.get("transport_error_owner"), max_len=80
        ),
        "transport_error_phase": _safe_text(
            diagnostics.get("transport_error_phase"), max_len=80
        ),
        "transport_error_status_code": _optional_int_text(
            diagnostics.get("transport_error_status_code")
        ),
        "transport_end_trigger": _safe_text(
            diagnostics.get("transport_end_trigger")
        ),
        "local_end_origin": _safe_text(diagnostics.get("local_end_origin")),
        "exception_type": _safe_text(
            diagnostics.get("exception_type"), max_len=80
        ),
        "exception_origin": _safe_text(
            diagnostics.get("exception_origin"), max_len=80
        ),
        "exception_errno_name": _safe_text(
            diagnostics.get("exception_errno_name"), max_len=80
        ),
        "httpcore_body_read_failure_type": _safe_text(
            diagnostics.get("httpcore_body_read_failure_type"), max_len=80
        ),
        "httpcore_response_close_trigger": _safe_text(
            diagnostics.get("httpcore_response_close_trigger")
        ),
        "downstream_usage_observer_status": _safe_text(
            diagnostics.get("downstream_usage_observer_status")
        ),
        "downstream_usage_observer_error_type": _safe_text(
            diagnostics.get("downstream_usage_observer_error_type"), max_len=80
        ),
        "downstream_usage_observer_abort_reason": _safe_text(
            diagnostics.get("downstream_usage_observer_abort_reason"), max_len=80
        ),
        "downstream_usage_completeness": _safe_text(
            diagnostics.get("downstream_usage_completeness")
        ),
        "downstream_failure_outcome": _safe_text(
            diagnostics.get("downstream_failure_outcome")
        ),
        "response_start_asgi_write_outcome": _safe_text(
            diagnostics.get("response_start_asgi_write_outcome")
        ),
        "response_start_asgi_write_error_type": _safe_text(
            diagnostics.get("response_start_asgi_write_error_type"), max_len=80
        ),
        "downstream_final_body_outcome": _safe_text(
            diagnostics.get("downstream_final_body_outcome")
        ),
        "downstream_final_body_error_type": _safe_text(
            diagnostics.get("downstream_final_body_error_type"), max_len=80
        ),
        "downstream_final_body_skip_reason": _safe_text(
            diagnostics.get("downstream_final_body_skip_reason"), max_len=80
        ),
        "cleanup_owner": _safe_text(diagnostics.get("cleanup_owner")),
        "cleanup_trigger": _safe_text(diagnostics.get("cleanup_trigger")),
        "cleanup_method": _safe_text(diagnostics.get("cleanup_method")),
        "cleanup_result": _safe_text(diagnostics.get("cleanup_result")),
        "cleanup_transport_action_actor": _safe_text(
            diagnostics.get("cleanup_transport_action_actor")
        ),
        "cleanup_transport_result_actor": _safe_text(
            diagnostics.get("cleanup_transport_result_actor")
        ),
        "cleanup_context_exit_actor": _safe_text(
            diagnostics.get("cleanup_context_exit_actor")
        ),
        "cleanup_failure_stage": _safe_text(
            diagnostics.get("cleanup_failure_stage")
        ),
        "pool_sweeper_trigger": _safe_text(
            diagnostics.get("pool_sweeper_trigger")
        ),
    }

    bool_fields = (
        "provider_penalty_eligible",
        "local_overload",
        "transport_metadata_available",
        "response_start_asgi_write_attempted",
        "response_start_asgi_write_completed",
        "upstream_eof_seen",
        "terminal_frame_seen",
        "terminal_frame_structured",
        "terminal_semantics_consistent",
        "upstream_terminal_seen",
        "upstream_terminal_validated",
        "response_completed_validated",
        "response_incomplete_validated",
        "failure_terminal_validated",
        "usage_object_seen",
        "usage_counters_seen",
        "usage_input_known",
        "usage_output_known",
        "usage_total_known",
        "usage_values_valid",
        "usage_alias_consistent",
        "usage_seen",
        "ember_queue_terminal_handoff_completed",
        "downstream_terminal_seen",
        "downstream_terminal_asgi_write_completed",
        "error_event_seen",
        "downstream_usage_object_seen",
        "downstream_usage_counters_seen",
        "downstream_usage_input_known",
        "downstream_usage_output_known",
        "downstream_usage_total_known",
        "downstream_usage_values_valid",
        "downstream_usage_alias_consistent",
        "downstream_usage_seen",
        "downstream_final_body_attempted",
        "downstream_final_body_completed",
        "local_cleanup_claimed_before_body_read_failure",
        "httpcore_events_truncated",
        "cleanup_transport_evicted",
        "cleanup_transport_isolated",
        "cleanup_transport_safe",
        "cleanup_context_exit_succeeded",
        "cleanup_detached_cleanup",
        "cleanup_actions_truncated",
        "cleanup_failure",
        "pool_sweeper_close_observed",
        "pool_sweeper_close_succeeded",
        "oaix_terminal_flush_marker_expected",
        "oaix_terminal_flush_marker_seen",
        "oaix_terminal_flush_marker_valid",
        "oaix_terminal_flush_marker_hash_matched",
        "oaix_terminal_flush_to_ember_receive_observed",
        "oaix_terminal_flush_marker_missing",
        "oaix_terminal_flush_lag_clamped_for_clock_order",
        "terminal_timeline_complete",
        "phase_sampled",
    )
    for key in bool_fields:
        attrs[key] = _bool_text(_safe_bool(diagnostics.get(key)))

    int_fields = (
        "httpcore_stream_id",
        "upstream_body_bytes",
        "upstream_chunk_count",
        "complete_event_count",
        "last_event_ordinal",
        "last_event_bytes",
        "partial_event_bytes",
        "declared_terminal_ordinal",
        "declared_terminal_bytes",
        "semantic_terminal_bytes",
        "semantic_terminal_sequence_number",
        "exception_errno",
        "exception_chain_depth",
        "cleanup_attempt_count",
        "oaix_terminal_flush_marker_schema_version",
        "oaix_terminal_flush_attempted_unix_nano",
        "oaix_terminal_flush_completed_unix_nano",
        "terminal_timeline_schema_version",
        "phase_sample_rate",
        "socket_unread_sample_rate",
        "socket_unread_samples",
        "socket_unread_bytes_total",
        "socket_unread_bytes_max",
        "socket_unread_bytes_last",
        "socket_unread_sample_failures",
    )
    for key in int_fields:
        attrs[key] = _optional_int_text(diagnostics.get(key))
    for point in _TERMINAL_TIMELINE_POINTS:
        for suffix in ("unix_nano", "monotonic_nano"):
            key = f"terminal_{point}_{suffix}"
            attrs[key] = _optional_int_text(diagnostics.get(key))
    for phase in _WORKER_PERFORMANCE_PHASES:
        if phase == "idempotency_hash":
            continue
        for suffix in ("samples", "wall_ns", "cpu_ns", "bytes", "events"):
            key = f"phase_{phase}_{suffix}"
            attrs[key] = _optional_int_text(diagnostics.get(key))

    text_fields = (
        "last_event_type",
        "last_event_sha256",
        "partial_event_sha256",
        "hash_scope",
        "event_hash_policy",
        "partial_hash_scope",
        "declared_terminal_sha256",
        "semantic_terminal_sha256",
        "first_upstream_body_at",
        "response_start_asgi_write_attempted_at",
        "response_start_asgi_write_completed_at",
        "response_start_asgi_write_error_at",
        "last_event_received_at",
        "declared_terminal_received_at",
        "terminal_frame_structured_at",
        "semantic_terminal_classified_at",
        "upstream_eof_at",
        "exception_at",
        "local_end_at",
        "ember_queue_terminal_handoff_completed_at",
        "downstream_terminal_asgi_write_completed_at",
        "downstream_failure_at",
        "downstream_usage_observer_aborted_at",
        "downstream_final_body_attempted_at",
        "downstream_final_body_completed_at",
        "downstream_final_body_error_at",
        "httpcore_body_read_failed_at",
        "httpcore_body_read_cancelled_at",
        "httpcore_response_close_started_at",
        "httpcore_response_close_completed_at",
        "httpcore_response_close_failed_at",
        "cleanup_started_at",
        "cleanup_completed_at",
        "pool_sweeper_close_started_at",
        "pool_sweeper_close_completed_at",
        "oaix_terminal_flush_marker_contract",
        "oaix_terminal_flush_marker_wire_sha256",
        "oaix_terminal_flush_marker_received_at",
        "oaix_terminal_flush_marker_invalid_reason",
        "oaix_terminal_flush_marker_missing_reason",
        "oaix_terminal_flush_marker_missing_at",
        "oaix_terminal_flush_attempted_at",
        "oaix_terminal_flush_completed_at",
        "terminal_timeline_clock",
        "terminal_received_semantics",
        "terminal_timeline_error",
        "phase_cpu_semantics",
        "phase_bytes_semantics",
        "phase_sampler_error",
        "socket_unread_semantics",
        "socket_unread_observer_error",
    )
    for key in text_fields:
        attrs[key] = _safe_text(diagnostics.get(key))

    for key in (
        "oaix_terminal_flush_duration_ms",
        "oaix_terminal_flush_to_ember_receive_signed_ms",
        "oaix_terminal_flush_attempt_to_ember_receive_ms",
        "oaix_terminal_flush_to_ember_receive_ms",
    ):
        attrs[key] = _optional_float_text(diagnostics.get(key))
    for point in _TERMINAL_TIMELINE_POINTS:
        key = f"terminal_{point}_from_receive_us"
        attrs[key] = _optional_float_text(diagnostics.get(key))
    for start, end in _TERMINAL_TIMELINE_TRANSITIONS:
        key = f"terminal_{start}_to_{end}_us"
        attrs[key] = _optional_float_text(diagnostics.get(key))

    attrs.update(
        _diagnostic_json_attrs(
            "terminal_semantics_inconsistency",
            diagnostics.get("terminal_semantics_inconsistency"),
        )
    )
    attrs.update(
        _diagnostic_json_attrs(
            "exception_chain",
            diagnostics.get("exception_chain"),
            source_truncated=bool(diagnostics.get("exception_chain_truncated")),
        )
    )
    attrs.update(
        _diagnostic_json_attrs(
            "httpcore_events",
            diagnostics.get("httpcore_events"),
            source_truncated=bool(diagnostics.get("httpcore_events_truncated")),
        )
    )
    attrs.update(
        _diagnostic_json_attrs(
            "cleanup_actions",
            diagnostics.get("cleanup_actions"),
            source_truncated=bool(diagnostics.get("cleanup_actions_truncated")),
        )
    )
    return _drop_empty(attrs)


def _is_stream_failure(current_info: dict[str, Any]) -> bool:
    outcome = _safe_text(current_info.get("stream_outcome")) or ""
    return bool(
        outcome
        and outcome not in {"completed", "downstream_disconnected"}
    )


def _base_attrs(
    *,
    service_name: str,
    service_version: str | None,
    identity_attrs: dict[str, str] | None,
    trace_id: str | None,
    request_id: str | None,
    parent_span_id: str | None,
    endpoint: str | None,
    method: str | None,
    path_template: str | None,
    route_id: str | None,
    model: str | None,
    provider: str | None,
    role: str | None,
    is_stream: bool | None,
    status_code: int,
    error_type: str | None,
    retry_count: int,
    cooldown_count: int,
    api_key_hash: str | None,
) -> dict[str, str]:
    return _drop_empty(
        {
            **(identity_attrs or {}),
            "service": service_name,
            "component": service_name,
            "service_version": _safe_text(service_version),
            "trace_id": _safe_text(trace_id),
            "request_id": _safe_text(request_id),
            "parent_span_id": _safe_text(parent_span_id),
            "route": _safe_text(endpoint),
            "route_id": route_id,
            "path_template": _safe_text(path_template or endpoint),
            "method": _safe_text(method),
            "request_kind": _safe_text(path_template or endpoint),
            "model": _safe_text(model),
            "provider": _safe_text(provider),
            "channel": _safe_text(provider),
            "role": _safe_text(role),
            "stream": _bool_text(is_stream),
            "streaming": _bool_text(is_stream),
            "status_code": _int_text(status_code),
            "status_class": _status_class(status_code),
            "error_type": error_type,
            "retry_count": _int_text(retry_count),
            "cooldown_count": _int_text(cooldown_count),
            "api_key_hash": api_key_hash,
        }
    )


def _identity_attrs_from_env() -> dict[str, str]:
    env_map = {
        "tenant_id": "FUGUE_OBSERVABILITY_TENANT_ID",
        "project_id": "FUGUE_OBSERVABILITY_PROJECT_ID",
        "app_id": "FUGUE_OBSERVABILITY_APP_ID",
        "runtime_id": "FUGUE_OBSERVABILITY_RUNTIME_ID",
        "pod": "HOSTNAME",
    }
    return _drop_empty({key: _env_text(env_name) for key, env_name in env_map.items()})


def _duration_ms_from_info(current_info: dict[str, Any]) -> int | None:
    process_time = current_info.get("process_time")
    try:
        if process_time is not None:
            return max(0, int(round(float(process_time) * 1000)))
    except (TypeError, ValueError):
        pass
    started_at = current_info.get("start_time")
    try:
        if started_at is not None:
            return max(0, int(round((time() - float(started_at)) * 1000)))
    except (TypeError, ValueError):
        pass
    return None


def _ttft_ms(spans: dict[str, Any]) -> int | None:
    value = _span_ms(spans, "upstream_first_chunk")
    if value > 0:
        return value
    value = _span_ms(spans, "upstream_headers_received")
    return value if value > 0 else None


def _stage_delta_ms(spans: dict[str, Any], stage: str, previous_stage: str) -> int:
    current = _span_ms(spans, stage)
    if current <= 0:
        return 0
    previous = _span_ms(spans, previous_stage)
    return current if previous <= 0 else max(0, current - previous)


def _runtime_int(runtime_metrics: dict[str, Any] | None, key: str) -> int | None:
    if not runtime_metrics:
        return None
    value = runtime_metrics.get(key)
    if value is None:
        return None
    return _safe_int(value, 0)


def _span_ms(spans: dict[str, Any], name: str) -> int:
    value = spans.get(name)
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return 0


def _split_endpoint(endpoint: str | None) -> tuple[str | None, str | None]:
    text = _safe_text(endpoint)
    if not text:
        return None, None
    parts = text.split(" ", 1)
    if len(parts) == 2 and parts[0].isalpha():
        return parts[0].upper(), parts[1].strip() or None
    return None, text


def _route_id(endpoint: str | None) -> str | None:
    _, path = _split_endpoint(endpoint)
    if not path:
        return None
    route = path.split("?", 1)[0].strip().rstrip("/") or "/"
    return route[:160]


def _endpoint_url(endpoint: str, path: str) -> str:
    base = endpoint.strip().rstrip("/")
    if base.endswith(("/v1/logs", "/v1/metrics", "/v1/traces")):
        base = base.rsplit("/v1/", 1)[0]
    return base + path


def _env_text(name: str) -> str | None:
    value = str(os.getenv(name, "")).strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, "")).strip() or default)
    except ValueError:
        return default


def _safe_text(value: Any, *, max_len: int = 256) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_len]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _bool_text(value: bool | None) -> str | None:
    if value is None:
        return None
    return "true" if value else "false"


def _int_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(max(0, int(value)))
    except (TypeError, ValueError):
        return None


def _optional_int_text(value: Any) -> str | None:
    if value is None:
        return None
    return _int_text(value)


def _transport_phase_attrs(
    primary: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    attrs: dict[str, str | None] = {}
    fallback = fallback if isinstance(fallback, dict) else {}
    for key in _TRANSPORT_PHASE_FIELDS:
        value = primary.get(key)
        if value is None:
            value = fallback.get(key)
        if value is None:
            continue
        if key.endswith(("_ms", "_bytes", "_count")):
            attrs[key] = _optional_int_text(value)
        else:
            attrs[key] = _safe_text(value, max_len=64)
    return attrs


def _iso_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _status_class(status_code: int) -> str:
    if status_code <= 0:
        return "unknown"
    return f"{status_code // 100}xx"


def _event_level(status_code: int) -> str:
    if status_code >= 500:
        return "error"
    if status_code >= 400:
        return "warning"
    return "info"


def _classify_error(status_code: int) -> str | None:
    if status_code <= 0 or status_code < 400:
        return None
    if status_code == 499:
        return "client_closed"
    if status_code == 429:
        return "rate_limited"
    if 400 <= status_code < 500:
        return "client_error"
    if status_code >= 500:
        return "upstream_or_server_error"
    return "error"


def _secret_hash(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _span_id(trace_id: str | None, request_id: str | None, stage: str) -> str:
    seed = "|".join([_safe_text(trace_id) or "", _safe_text(request_id) or "", stage])
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _drop_empty(values: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        result[str(key)] = text
    return result
