from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
from dataclasses import dataclass, field
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


class FugueObservabilityClient:
    def __init__(self, config: FugueObservabilityConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._client: httpx.AsyncClient | None = None
        self._dropped = 0
        self._export_errors = 0

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
        if status_code < 400 and self.config.sample_rate < 1.0 and random.random() > self.config.sample_rate:
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
        if self.config.emit_stage_spans:
            self._emit_events(_TRACE_ENDPOINT, telemetry["traces"])
        if self.config.emit_metrics:
            self._emit_events(_METRIC_ENDPOINT, telemetry["metrics"])

    def _emit_events(self, path: str, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait((path, {"events": events}))
        except asyncio.QueueFull:
            self._dropped += len(events)
            if self._dropped == len(events) or self._dropped % 100 == 0:
                logger.warning("Fugue observability queue full; dropped %s event(s)", self._dropped)

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            path, payload = await self._queue.get()
            try:
                await self._post_json(path, payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._export_errors += 1
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
    retry_count = _safe_int(current_info.get("retry_count"), 0)
    cooldown_count = _safe_int(current_info.get("cooldown_count"), 0)
    is_stream = _safe_bool(current_info.get("stream"))
    api_key_hash = _secret_hash(current_info.get("api_key"))

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

    logs = [
        {
            "timestamp": _iso_timestamp(now),
            "level": _event_level(status_code),
            "service": service_name,
            "trace_id": trace_id,
            "request_id": request_id,
            "event": "request_summary",
            "event_type": "request_summary",
            "source": service_name,
            "message": "uni-api-ember request finished",
            "attributes": _drop_empty(
                {
                    **base,
                    "duration_ms": _int_text(duration_ms),
                    "total_ms": _int_text(duration_ms),
                    "ttfb_ms": _int_text(ttft_ms),
                    "ttft_ms": _int_text(ttft_ms),
                    "upstream_ms": _int_text(_stage_delta_ms(spans, "upstream_headers_received", "upstream_send_start")),
                    "status_class": _status_class(status_code),
                    "request_kind": _safe_text(current_info.get("request_kind")),
                }
            ),
            "summary": _drop_empty(
                {
                    "message_roles": _safe_text(current_info.get("message_roles")),
                    "role_counts": _safe_text(current_info.get("role_counts")),
                    "client_pool_wait_ms": _int_text(_span_ms(spans, "upstream_pool_wait_ms")),
                    "event_loop_lag_ms": _int_text(_runtime_int(runtime_metrics, "event_loop_lag_ms")),
                    "inflight_requests": _int_text(_runtime_int(runtime_metrics, "inflight_requests")),
                    "waiting_first_byte": _int_text(_runtime_int(runtime_metrics, "waiting_first_byte")),
                }
            ),
        }
    ]

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

    metrics = _request_metric_events(
        service_name=service_name,
        identity_attrs=identity_attrs,
        timestamp=now,
        method=method,
        status_code=status_code,
        route_id=route_id,
        values={
            "uniapi_ember_request_duration_ms": duration_ms,
            "uniapi_ember_request_ttfb_ms": ttft_ms,
            "uniapi_ember_inflight_requests": _runtime_int(runtime_metrics, "inflight_requests"),
            "uniapi_ember_waiting_first_byte": _runtime_int(runtime_metrics, "waiting_first_byte"),
            "uniapi_ember_event_loop_lag_ms": _runtime_int(runtime_metrics, "event_loop_lag_ms"),
            "uniapi_ember_client_pool_in_use": _runtime_int(runtime_metrics, "upstream_pool_in_use"),
            "uniapi_ember_client_pool_wait_ms": _span_ms(spans, "upstream_pool_wait_ms"),
            "uniapi_ember_retry_total": retry_count,
            "uniapi_ember_provider_cooldown_total": cooldown_count,
            "uniapi_ember_upstream_errors_total": 1 if status_code >= 500 else 0,
        },
    )
    return {"logs": logs, "traces": traces, "metrics": metrics}


def _stage_rows(spans: dict[str, Any], duration_ms: int | None) -> list[tuple[str, int, dict[str, str]]]:
    rows: list[tuple[str, int, dict[str, str]]] = []
    previous_stage = ""
    for stage in _STAGE_ORDER:
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
    base_attrs = _drop_empty(
        {
            **(identity_attrs or {}),
            "component": service_name,
            "route_id": route_id,
            "method": method,
            "status_class": _status_class(status_code),
        }
    )
    events = []
    for metric, value in values.items():
        if value is None:
            continue
        events.append(
            {
                "timestamp": _iso_timestamp(timestamp),
                "kind": "metric",
                "source": service_name,
                "message": metric,
                "metric": metric,
                "value": max(0, int(value)),
                "attributes": base_attrs,
            }
        )
    return events


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
