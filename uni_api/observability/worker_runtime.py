from __future__ import annotations

import asyncio
import math
import os
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from uni_api.observability.threadpool_tasks import (
    active_threadpool_task_categories,
    submit_threadpool_task,
    threadpool_task_snapshot,
)


TERMINAL_HOP_BUCKETS_MS = (
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
    30_000.0,
)

PERFORMANCE_PHASES = (
    "socket_receive",
    "sse_frame",
    "json_parse",
    "observer_hash",
    "queue_put",
    "asgi_write",
    "idempotency_hash",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _bounded_text(value: Any, *, limit: int = 256) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:limit]


class CumulativeHistogram:
    """A low-cardinality cumulative histogram owned by one worker process."""

    def __init__(self, bounds: tuple[float, ...]) -> None:
        normalized = tuple(sorted({float(value) for value in bounds if value >= 0}))
        if not normalized:
            raise ValueError("histogram bounds cannot be empty")
        self._bounds = normalized
        self._buckets = [0] * len(normalized)
        self._count = 0
        self._sum = 0.0

    def observe(self, value: float) -> bool:
        try:
            sample = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(sample) or sample < 0:
            return False
        self._count += 1
        self._sum += sample
        for index, bound in enumerate(self._bounds):
            if sample <= bound:
                self._buckets[index] += 1
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "bounds_ms": list(self._bounds),
            "cumulative_buckets": {
                _bucket_label(bound): self._buckets[index]
                for index, bound in enumerate(self._bounds)
            },
            "count": self._count,
            "sum_ms": self._sum,
            "infinite_bucket": self._count,
        }


def _bucket_label(bound: float) -> str:
    return str(int(bound)) if bound.is_integer() else str(bound)


class WorkerRuntimeObserver:
    """Measure one worker without blocking request or stream paths.

    SSE hot paths only increment integers. CPU/rate sampling and optional
    profiling run out of band. Export callbacks must be non-blocking enqueue
    operations; failures are isolated from the serving path.
    """

    def __init__(
        self,
        *,
        inflight_supplier: Callable[[], int] | None = None,
        snapshot_emitter: Callable[[dict[str, Any]], Any] | None = None,
        profile_emitter: Callable[[dict[str, Any]], Any] | None = None,
        terminal_hop_emitter: Callable[[dict[str, Any]], Any] | None = None,
        sample_interval_seconds: float | None = None,
        cpu_profile_enabled: bool | None = None,
        cpu_profile_trigger_cores: float | None = None,
        cpu_profile_trigger_samples: int | None = None,
        cpu_profile_duration_seconds: float | None = None,
        cpu_profile_sample_hz: float | None = None,
        cpu_profile_cooldown_seconds: float | None = None,
        phase_sample_rate: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        process_time: Callable[[], float] = time.process_time,
    ) -> None:
        self.worker_pid = os.getpid()
        pod = _bounded_text(os.getenv("HOSTNAME"), limit=128) or "local"
        self.worker_id = f"{pod}:{self.worker_pid}"
        self.source_revision = _bounded_text(
            os.getenv("FUGUE_SOURCE_COMMIT_SHA")
            or os.getenv("GIT_COMMIT")
            or os.getenv("SOURCE_COMMIT"),
            limit=64,
        )
        self.started_at = _utc_now()
        self._inflight_supplier = inflight_supplier or (lambda: 0)
        self._snapshot_emitter = snapshot_emitter
        self._profile_emitter = profile_emitter
        self._terminal_hop_emitter = terminal_hop_emitter
        self._monotonic = monotonic
        self._process_time = process_time
        self.sample_interval_seconds = sample_interval_seconds or _env_float(
            "WORKER_OBSERVABILITY_SAMPLE_INTERVAL_SECONDS",
            5.0,
            minimum=1.0,
            maximum=60.0,
        )
        self.cpu_profile_enabled = (
            _env_bool("WORKER_CPU_PROFILE_ENABLED", True)
            if cpu_profile_enabled is None
            else bool(cpu_profile_enabled)
        )
        self.cpu_profile_trigger_cores = (
            cpu_profile_trigger_cores
            if cpu_profile_trigger_cores is not None
            else _env_float(
                "WORKER_CPU_PROFILE_TRIGGER_CORES",
                0.9,
                minimum=0.1,
                maximum=16.0,
            )
        )
        self.cpu_profile_trigger_samples = (
            cpu_profile_trigger_samples
            if cpu_profile_trigger_samples is not None
            else _env_int(
                "WORKER_CPU_PROFILE_TRIGGER_SAMPLES",
                2,
                minimum=1,
                maximum=12,
            )
        )
        self.cpu_profile_duration_seconds = (
            cpu_profile_duration_seconds
            if cpu_profile_duration_seconds is not None
            else _env_float(
                "WORKER_CPU_PROFILE_DURATION_SECONDS",
                10.0,
                minimum=2.0,
                maximum=30.0,
            )
        )
        self.cpu_profile_sample_hz = (
            cpu_profile_sample_hz
            if cpu_profile_sample_hz is not None
            else _env_float(
                "WORKER_CPU_PROFILE_SAMPLE_HZ",
                5.0,
                minimum=1.0,
                maximum=20.0,
            )
        )
        self.cpu_profile_cooldown_seconds = (
            cpu_profile_cooldown_seconds
            if cpu_profile_cooldown_seconds is not None
            else _env_float(
                "WORKER_CPU_PROFILE_COOLDOWN_SECONDS",
                900.0,
                minimum=60.0,
                maximum=86_400.0,
            )
        )
        self.phase_sample_rate = (
            max(1, int(phase_sample_rate))
            if phase_sample_rate is not None
            else _env_int(
                "WORKER_PHASE_SAMPLE_RATE",
                128,
                minimum=1,
                maximum=65_536,
            )
        )

        self._sse_events_total = 0
        self._sse_bytes_total = 0
        self._terminal_hop_histogram = CumulativeHistogram(
            TERMINAL_HOP_BUCKETS_MS
        )
        self._terminal_hop_invalid_total = 0
        self._terminal_hop_marker_missing_total = 0
        self._last_wall = self._monotonic()
        self._last_cpu = self._process_time()
        self._last_sse_events = 0
        self._last_sse_bytes = 0
        self._cpu_cores = 0.0
        self._sse_events_per_second = 0.0
        self._sse_bytes_per_second = 0.0
        self._cpu_seconds_per_sse_mebibyte: float | None = None
        self._sample_elapsed_seconds = 0.0
        self._profile_trigger_streak = 0
        self._profile_trigger_total = 0
        self._profile_completed_total = 0
        self._profile_failed_total = 0
        self._profile_task: asyncio.Task[None] | None = None
        self._sampler_task: asyncio.Task[None] | None = None
        self._profile_stop = threading.Event()
        self._next_profile_after = 0.0
        self._latest_profile: dict[str, Any] | None = None
        self._phase_sample_sequences: Counter[str] = Counter()
        self._phase_metrics: dict[str, Counter[str]] = {
            phase: Counter() for phase in PERFORMANCE_PHASES
        }
        self._socket_unread_samples_total = 0
        self._socket_unread_bytes_total = 0
        self._socket_unread_bytes_max = 0
        self._socket_unread_bytes_last: int | None = None
        self._socket_unread_sample_failures_total = 0

    def set_inflight_supplier(self, supplier: Callable[[], int]) -> None:
        self._inflight_supplier = supplier

    def set_emitters(
        self,
        *,
        snapshot: Callable[[dict[str, Any]], Any] | None = None,
        profile: Callable[[dict[str, Any]], Any] | None = None,
        terminal_hop: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._snapshot_emitter = snapshot
        self._profile_emitter = profile
        self._terminal_hop_emitter = terminal_hop

    def record_sse_chunk(self, size: int) -> None:
        try:
            observed = int(size)
        except (TypeError, ValueError):
            return
        if observed > 0:
            self._sse_bytes_total += observed

    def record_sse_event(self, _size: int = 0) -> None:
        self._sse_events_total += 1

    def should_sample_phase_request(self, scope: str = "default") -> bool:
        normalized_scope = (
            scope
            if scope in {"default", "responses_stream", "idempotency_hash"}
            else "default"
        )
        sequence = self._phase_sample_sequences[normalized_scope]
        self._phase_sample_sequences[normalized_scope] += 1
        return sequence % self.phase_sample_rate == 0

    def record_phase_sample(
        self,
        phase: str,
        *,
        wall_ns: int,
        cpu_ns: int,
        bytes_count: int = 0,
        events: int = 0,
    ) -> bool:
        if phase not in self._phase_metrics:
            return False
        try:
            normalized_wall = max(0, int(wall_ns))
            normalized_cpu = max(0, int(cpu_ns))
            normalized_bytes = max(0, int(bytes_count))
            normalized_events = max(0, int(events))
        except (TypeError, ValueError):
            return False
        metrics = self._phase_metrics[phase]
        metrics["samples_total"] += 1
        metrics["wall_ns_total"] += normalized_wall
        metrics["cpu_ns_total"] += normalized_cpu
        metrics["bytes_total"] += normalized_bytes
        metrics["events_total"] += normalized_events
        return True

    def record_socket_unread_bytes(self, value: int | None) -> bool:
        if value is None:
            self._socket_unread_sample_failures_total += 1
            return False
        try:
            observed = max(0, int(value))
        except (TypeError, ValueError):
            self._socket_unread_sample_failures_total += 1
            return False
        self._socket_unread_samples_total += 1
        self._socket_unread_bytes_total += observed
        self._socket_unread_bytes_max = max(
            self._socket_unread_bytes_max,
            observed,
        )
        self._socket_unread_bytes_last = observed
        return True

    def record_terminal_hop(self, observation: dict[str, Any]) -> bool:
        try:
            lag_ms = float(observation.get("lag_ms"))
        except (AttributeError, TypeError, ValueError):
            self._terminal_hop_invalid_total += 1
            return False
        if not self._terminal_hop_histogram.observe(lag_ms):
            self._terminal_hop_invalid_total += 1
            return False
        emitter = self._terminal_hop_emitter
        if emitter is not None:
            try:
                emitter(dict(observation))
            except Exception:
                # Observability must never alter stream delivery.
                self._terminal_hop_invalid_total += 1
        return True

    def record_terminal_marker_missing(self) -> None:
        self._terminal_hop_marker_missing_total += 1

    async def start(self) -> None:
        if self._sampler_task is not None:
            return
        self._last_wall = self._monotonic()
        self._last_cpu = self._process_time()
        self._last_sse_events = self._sse_events_total
        self._last_sse_bytes = self._sse_bytes_total
        self._profile_stop.clear()
        self._sampler_task = asyncio.create_task(
            self._sample_loop(),
            name="uni-api-worker-runtime-observer",
        )

    async def stop(self) -> None:
        task = self._sampler_task
        self._sampler_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._profile_stop.set()
        profile_task = self._profile_task
        if profile_task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(profile_task),
                    timeout=min(2.0, self.cpu_profile_duration_seconds),
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    async def _sample_loop(self) -> None:
        while True:
            await asyncio.sleep(self.sample_interval_seconds)
            self.sample_now()

    def sample_now(
        self,
        *,
        wall_time: float | None = None,
        cpu_time: float | None = None,
        emit: bool = True,
        allow_profile: bool = True,
    ) -> dict[str, Any]:
        now_wall = self._monotonic() if wall_time is None else float(wall_time)
        now_cpu = self._process_time() if cpu_time is None else float(cpu_time)
        elapsed = max(0.0, now_wall - self._last_wall)
        cpu_delta = max(0.0, now_cpu - self._last_cpu)
        event_delta = max(0, self._sse_events_total - self._last_sse_events)
        byte_delta = max(0, self._sse_bytes_total - self._last_sse_bytes)
        self._last_wall = now_wall
        self._last_cpu = now_cpu
        self._last_sse_events = self._sse_events_total
        self._last_sse_bytes = self._sse_bytes_total
        self._sample_elapsed_seconds = elapsed
        if elapsed > 0:
            self._cpu_cores = cpu_delta / elapsed
            self._sse_events_per_second = event_delta / elapsed
            self._sse_bytes_per_second = byte_delta / elapsed
        if byte_delta > 0:
            mebibytes = byte_delta / float(1024 * 1024)
            self._cpu_seconds_per_sse_mebibyte = cpu_delta / mebibytes
        else:
            self._cpu_seconds_per_sse_mebibyte = None

        if self._cpu_cores >= self.cpu_profile_trigger_cores:
            self._profile_trigger_streak += 1
        else:
            self._profile_trigger_streak = 0

        snapshot = self.snapshot(cpu_seconds_total=now_cpu)
        if emit and self._snapshot_emitter is not None:
            try:
                self._snapshot_emitter(snapshot)
            except Exception:
                pass
        if allow_profile:
            self._maybe_start_profile(now_wall)
        return snapshot

    def _maybe_start_profile(self, now_wall: float) -> None:
        if not self.cpu_profile_enabled or not sys.platform.startswith("linux"):
            return
        if self._profile_trigger_streak < self.cpu_profile_trigger_samples:
            return
        if now_wall < self._next_profile_after:
            return
        if self._profile_task is not None and not self._profile_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._profile_trigger_streak = 0
        self._profile_trigger_total += 1
        self._next_profile_after = now_wall + self.cpu_profile_cooldown_seconds
        trigger_cpu_cores = self._cpu_cores
        self._profile_task = loop.create_task(
            self._run_profile(trigger_cpu_cores),
            name="uni-api-worker-on-cpu-profile",
        )

    async def _run_profile(self, trigger_cpu_cores: float) -> None:
        collector = LinuxThreadOnCPUProfiler(
            worker_id=self.worker_id,
            source_revision=self.source_revision,
            duration_seconds=self.cpu_profile_duration_seconds,
            sample_hz=self.cpu_profile_sample_hz,
            stop_event=self._profile_stop,
        )
        try:
            ticket = submit_threadpool_task("on_cpu_profile")
            result = await asyncio.to_thread(
                ticket.run,
                collector.run,
                trigger_cpu_cores=trigger_cpu_cores,
            )
            self._latest_profile = result
            if result.get("status") == "completed":
                self._profile_completed_total += 1
            else:
                self._profile_failed_total += 1
            if self._profile_emitter is not None:
                self._profile_emitter(dict(result))
        except asyncio.CancelledError:
            ticket.cancel_if_queued()
            raise
        except Exception as exc:
            self._profile_failed_total += 1
            self._latest_profile = {
                "schema_version": 1,
                "profile_id": uuid.uuid4().hex,
                "worker_id": self.worker_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "finished_at": _utc_now(),
            }
            if self._profile_emitter is not None:
                try:
                    self._profile_emitter(dict(self._latest_profile))
                except Exception:
                    pass

    def snapshot(self, *, cpu_seconds_total: float | None = None) -> dict[str, Any]:
        try:
            inflight = max(0, int(self._inflight_supplier()))
        except Exception:
            inflight = 0
        histogram = self._terminal_hop_histogram.snapshot()
        phase_samples: dict[str, dict[str, int | float]] = {}
        for phase in PERFORMANCE_PHASES:
            metrics = self._phase_metrics[phase]
            samples = int(metrics.get("samples_total") or 0)
            if samples <= 0:
                continue
            events = int(metrics.get("events_total") or 0)
            wall_ns = int(metrics.get("wall_ns_total") or 0)
            cpu_ns = int(metrics.get("cpu_ns_total") or 0)
            row: dict[str, int | float] = {
                "samples_total": samples,
                "wall_ns_total": wall_ns,
                "cpu_ns_total": cpu_ns,
                "bytes_total": int(metrics.get("bytes_total") or 0),
                "events_total": events,
            }
            if events > 0:
                row["wall_us_per_event"] = wall_ns / (events * 1000.0)
                row["cpu_us_per_event"] = cpu_ns / (events * 1000.0)
            phase_samples[phase] = row
        phase_sampling = {
            scope: {
                "candidates_total": candidates,
                "selected_total": (
                    (candidates + self.phase_sample_rate - 1)
                    // self.phase_sample_rate
                ),
            }
            for scope, candidates in sorted(
                self._phase_sample_sequences.items()
            )
            if candidates > 0
        }
        return {
            "worker_metrics_schema_version": 2,
            "worker_id": self.worker_id,
            "worker_pid": self.worker_pid,
            "worker_started_at": self.started_at,
            "worker_source_revision": self.source_revision,
            "worker_cpu_seconds_total": (
                self._process_time()
                if cpu_seconds_total is None
                else max(0.0, float(cpu_seconds_total))
            ),
            "worker_cpu_cores": self._cpu_cores,
            "worker_single_core_saturation_ratio": self._cpu_cores,
            "worker_sse_events_total": self._sse_events_total,
            "worker_sse_bytes_total": self._sse_bytes_total,
            "worker_sse_events_per_second": self._sse_events_per_second,
            "worker_sse_bytes_per_second": self._sse_bytes_per_second,
            "worker_inflight_requests": inflight,
            "worker_cpu_seconds_per_sse_mebibyte": (
                self._cpu_seconds_per_sse_mebibyte
            ),
            "worker_metrics_sample_elapsed_seconds": self._sample_elapsed_seconds,
            "worker_phase_sample_rate": self.phase_sample_rate,
            "worker_phase_sampling": phase_sampling,
            "worker_phase_samples": phase_samples,
            "worker_threadpool_tasks": threadpool_task_snapshot(),
            "worker_socket_unread_samples_total": (
                self._socket_unread_samples_total
            ),
            "worker_socket_unread_bytes_total": self._socket_unread_bytes_total,
            "worker_socket_unread_bytes_max": self._socket_unread_bytes_max,
            "worker_socket_unread_bytes_last": self._socket_unread_bytes_last,
            "worker_socket_unread_sample_failures_total": (
                self._socket_unread_sample_failures_total
            ),
            "worker_cpu_profile_enabled": self.cpu_profile_enabled,
            "worker_cpu_profile_trigger_cores": self.cpu_profile_trigger_cores,
            "worker_cpu_profile_trigger_samples": self.cpu_profile_trigger_samples,
            "worker_cpu_profile_trigger_streak": self._profile_trigger_streak,
            "worker_cpu_profile_running": bool(
                self._profile_task is not None and not self._profile_task.done()
            ),
            "worker_cpu_profile_trigger_total": self._profile_trigger_total,
            "worker_cpu_profile_completed_total": self._profile_completed_total,
            "worker_cpu_profile_failed_total": self._profile_failed_total,
            "worker_cpu_profile_latest": self._latest_profile,
            "oaix_terminal_flush_to_ember_receive_histogram": histogram,
            "oaix_terminal_flush_to_ember_receive_invalid_total": (
                self._terminal_hop_invalid_total
            ),
            "oaix_terminal_flush_marker_missing_total": (
                self._terminal_hop_marker_missing_total
            ),
        }


class LinuxThreadOnCPUProfiler:
    """Low-frequency statistical profiler using per-thread Linux CPU ticks.

    It runs in a helper thread, reads only ``/proc/self/task/*/stat``, and
    samples Python stacks without tracing calls or request data. CPU deltas are
    attributed to the stack observed for that native thread in the interval.
    """

    def __init__(
        self,
        *,
        worker_id: str,
        source_revision: str | None,
        duration_seconds: float,
        sample_hz: float,
        stop_event: threading.Event,
    ) -> None:
        self.worker_id = worker_id
        self.source_revision = source_revision
        self.duration_seconds = max(0.1, float(duration_seconds))
        self.sample_hz = max(0.5, float(sample_hz))
        self.stop_event = stop_event

    def run(self, *, trigger_cpu_cores: float) -> dict[str, Any]:
        profile_id = uuid.uuid4().hex
        started_at = _utc_now()
        started = time.monotonic()
        interval = 1.0 / self.sample_hz
        profiler_native_id = threading.get_native_id()
        previous = _read_thread_cpu_ticks()
        stack_ticks: Counter[tuple[str, ...]] = Counter()
        leaf_ticks: Counter[str] = Counter()
        stack_samples: Counter[tuple[str, ...]] = Counter()
        task_category_ticks: Counter[str] = Counter()
        task_category_samples: Counter[str] = Counter()
        task_category_source_ticks: Counter[tuple[str, str]] = Counter()
        task_category_source_samples: Counter[tuple[str, str]] = Counter()
        sample_rounds = 0
        active_thread_samples = 0
        read_errors = 0

        while time.monotonic() - started < self.duration_seconds:
            remaining = self.duration_seconds - (time.monotonic() - started)
            if remaining <= 0 or self.stop_event.wait(min(interval, remaining)):
                break
            try:
                current = _read_thread_cpu_ticks()
                frames = _native_thread_frames()
                thread_names = _native_thread_names()
                task_categories = active_threadpool_task_categories()
            except Exception:
                read_errors += 1
                continue
            sample_rounds += 1
            for native_id, ticks in current.items():
                if native_id == profiler_native_id:
                    continue
                delta = max(0, ticks - previous.get(native_id, ticks))
                if delta <= 0:
                    continue
                stack = _frame_stack(frames.get(native_id))
                if not stack:
                    stack = (f"native-thread:{native_id}",)
                stack_ticks[stack] += delta
                stack_samples[stack] += 1
                leaf_ticks[stack[-1]] += delta
                category = task_categories.get(native_id)
                category_source = "explicit_task_tag" if category else None
                if not category:
                    category, category_source = _threadpool_category_from_sample(
                        thread_names.get(native_id),
                        stack,
                    )
                if category:
                    task_category_ticks[category] += delta
                    task_category_samples[category] += 1
                    assert category_source is not None
                    source_key = (category, category_source)
                    task_category_source_ticks[source_key] += delta
                    task_category_source_samples[source_key] += 1
                active_thread_samples += 1
            previous = current

        finished_at = _utc_now()
        ticks_per_second = _clock_ticks_per_second()
        total_ticks = sum(stack_ticks.values())
        top_stacks = [
            {
                "stack": list(stack),
                "cpu_ticks": ticks,
                "cpu_seconds": ticks / ticks_per_second,
                "samples": stack_samples[stack],
            }
            for stack, ticks in stack_ticks.most_common(20)
        ]
        top_leaf_functions = [
            {
                "function": function,
                "cpu_ticks": ticks,
                "cpu_seconds": ticks / ticks_per_second,
            }
            for function, ticks in leaf_ticks.most_common(20)
        ]
        threadpool_categories = [
            {
                "category": category,
                "cpu_ticks": ticks,
                "cpu_seconds": ticks / ticks_per_second,
                "samples": task_category_samples[category],
                "sources": [
                    {
                        "source": source,
                        "cpu_ticks": source_ticks,
                        "cpu_seconds": source_ticks / ticks_per_second,
                        "samples": task_category_source_samples[
                            (category, source)
                        ],
                    }
                    for (source_category, source), source_ticks in (
                        task_category_source_ticks.most_common()
                    )
                    if source_category == category
                ],
            }
            for category, ticks in task_category_ticks.most_common(20)
        ]
        return {
            "schema_version": 2,
            "profile_id": profile_id,
            "worker_id": self.worker_id,
            "source_revision": self.source_revision,
            "status": "completed" if sample_rounds > 0 else "no_samples",
            "trigger_cpu_cores": max(0.0, float(trigger_cpu_cores)),
            "started_at": started_at,
            "finished_at": finished_at,
            "configured_duration_seconds": self.duration_seconds,
            "observed_duration_seconds": max(0.0, time.monotonic() - started),
            "sample_hz": self.sample_hz,
            "sample_rounds": sample_rounds,
            "active_thread_samples": active_thread_samples,
            "profiled_cpu_ticks": total_ticks,
            "profiled_cpu_seconds": total_ticks / ticks_per_second,
            "proc_read_errors": read_errors,
            "threadpool_classification_semantics": (
                "explicit_tag_then_dedicated_name_then_bounded_stack_v1"
            ),
            "top_leaf_functions": top_leaf_functions,
            "top_stacks": top_stacks,
            "threadpool_categories": threadpool_categories,
        }


def _clock_ticks_per_second() -> float:
    try:
        return max(1.0, float(os.sysconf("SC_CLK_TCK")))
    except (AttributeError, OSError, TypeError, ValueError):
        return 100.0


def _read_thread_cpu_ticks() -> dict[int, int]:
    result: dict[int, int] = {}
    task_root = Path("/proc/self/task")
    for entry in task_root.iterdir():
        try:
            native_id = int(entry.name)
            raw = (entry / "stat").read_text(encoding="ascii", errors="replace")
            closing = raw.rfind(")")
            if closing < 0:
                continue
            fields = raw[closing + 2 :].split()
            # fields[0] is stat field 3 (state); utime/stime are 14/15.
            user_ticks = int(fields[11])
            system_ticks = int(fields[12])
            result[native_id] = max(0, user_ticks + system_ticks)
        except (FileNotFoundError, IndexError, IsADirectoryError, OSError, ValueError):
            continue
    return result


def _native_thread_frames() -> dict[int, Any]:
    by_ident = sys._current_frames()
    result: dict[int, Any] = {}
    for thread in threading.enumerate():
        if thread.native_id is None or thread.ident is None:
            continue
        frame = by_ident.get(thread.ident)
        if frame is not None:
            result[int(thread.native_id)] = frame
    return result


def _native_thread_names() -> dict[int, str]:
    return {
        int(thread.native_id): str(thread.name)
        for thread in threading.enumerate()
        if thread.native_id is not None
    }


def _threadpool_category_from_sample(
    thread_name: str | None,
    stack: tuple[str, ...],
) -> tuple[str | None, str | None]:
    name = str(thread_name or "")
    dedicated_prefixes = (
        ("uni-api-json", "json_parse"),
        ("uni-api-upstream-body", "upstream_response_decode"),
        ("uni-api-idempotency-spool", "idempotency_spool"),
        ("uni-api-body", "request_body_decode"),
    )
    for prefix, category in dedicated_prefixes:
        if name.startswith(prefix):
            return category, "dedicated_thread_name"
    if name.startswith(("asyncio_", "ThreadPoolExecutor-")):
        if any(
            row.startswith(("json/__init__.py:", "json/encoder.py:"))
            for row in stack
        ):
            return "json_serialization", "default_executor_stack"
        return "other", "default_executor_stack"
    return None, None


def _frame_stack(frame: Any, *, max_depth: int = 24) -> tuple[str, ...]:
    rows: list[str] = []
    current = frame
    while current is not None and len(rows) < max_depth:
        code = getattr(current, "f_code", None)
        if code is None:
            break
        rows.append(
            f"{_safe_code_path(code.co_filename)}:{code.co_name}:{current.f_lineno}"
        )
        current = current.f_back
    rows.reverse()
    return tuple(rows)


def _safe_code_path(filename: str) -> str:
    normalized = str(filename or "").replace("\\", "/")
    for marker in ("/uni_api/", "/core/"):
        if marker in normalized:
            return marker.strip("/") + "/" + normalized.split(marker, 1)[1]
    parts = [part for part in normalized.split("/") if part]
    return "/".join(parts[-2:]) if parts else "unknown"
