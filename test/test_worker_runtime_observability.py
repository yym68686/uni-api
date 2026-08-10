import asyncio
import math

from uni_api.observability import worker_runtime
from uni_api.observability.worker_runtime import (
    CumulativeHistogram,
    WorkerRuntimeObserver,
)


def test_cumulative_histogram_records_inclusive_buckets():
    histogram = CumulativeHistogram((10, 100, 1000))

    assert histogram.observe(10)
    assert histogram.observe(50)
    assert histogram.observe(500)
    assert not histogram.observe(-1)
    assert not histogram.observe(float("nan"))

    snapshot = histogram.snapshot()
    assert snapshot["count"] == 3
    assert snapshot["sum_ms"] == 560
    assert snapshot["cumulative_buckets"] == {
        "10": 1,
        "100": 2,
        "1000": 3,
    }
    assert snapshot["infinite_bucket"] == 3


def test_worker_sample_reports_cpu_stream_rates_inflight_and_cpu_per_mib():
    emitted = []
    wall = [100.0]
    cpu = [20.0]
    observer = WorkerRuntimeObserver(
        inflight_supplier=lambda: 7,
        snapshot_emitter=emitted.append,
        cpu_profile_enabled=False,
        monotonic=lambda: wall[0],
        process_time=lambda: cpu[0],
    )
    observer.record_sse_chunk(2 * 1024 * 1024)
    observer.record_sse_event(100)
    observer.record_sse_event(100)

    wall[0] = 105.0
    cpu[0] = 24.5
    snapshot = observer.sample_now()

    assert math.isclose(snapshot["worker_cpu_cores"], 0.9)
    assert math.isclose(snapshot["worker_sse_events_per_second"], 0.4)
    assert math.isclose(
        snapshot["worker_sse_bytes_per_second"],
        (2 * 1024 * 1024) / 5,
    )
    assert math.isclose(
        snapshot["worker_cpu_seconds_per_sse_mebibyte"],
        2.25,
    )
    assert snapshot["worker_inflight_requests"] == 7
    assert emitted == [snapshot]


def test_terminal_hop_observation_updates_histogram_and_fail_open_emitter():
    emitted = []
    observer = WorkerRuntimeObserver(
        terminal_hop_emitter=emitted.append,
        cpu_profile_enabled=False,
    )

    assert observer.record_terminal_hop(
        {
            "lag_ms": 7654.25,
            "request_id": "request-safe",
            "terminal_wire_sha256": "a" * 64,
        }
    )
    assert not observer.record_terminal_hop({"lag_ms": -3})

    snapshot = observer.snapshot()
    histogram = snapshot["oaix_terminal_flush_to_ember_receive_histogram"]
    assert histogram["count"] == 1
    assert histogram["cumulative_buckets"]["5000"] == 0
    assert histogram["cumulative_buckets"]["10000"] == 1
    assert snapshot["oaix_terminal_flush_to_ember_receive_invalid_total"] == 1
    assert emitted[0]["lag_ms"] == 7654.25


def test_phase_sampling_is_exact_and_worker_snapshot_is_bounded():
    observer = WorkerRuntimeObserver(
        cpu_profile_enabled=False,
        phase_sample_rate=4,
    )

    assert [observer.should_sample_phase_request() for _ in range(8)] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert observer.should_sample_phase_request("responses_stream") is True
    assert observer.should_sample_phase_request("idempotency_hash") is True
    assert observer.record_phase_sample(
        "json_parse",
        wall_ns=4_000,
        cpu_ns=2_000,
        bytes_count=512,
        events=2,
    )
    assert not observer.record_phase_sample(
        "unbounded-user-controlled-phase",
        wall_ns=1,
        cpu_ns=1,
    )
    assert observer.record_socket_unread_bytes(4096)
    assert not observer.record_socket_unread_bytes(None)

    snapshot = observer.snapshot(cpu_seconds_total=1.0)
    assert snapshot["worker_metrics_schema_version"] == 2
    assert snapshot["worker_phase_sample_rate"] == 4
    assert snapshot["worker_phase_sampling"] == {
        "default": {"candidates_total": 8, "selected_total": 2},
        "idempotency_hash": {"candidates_total": 1, "selected_total": 1},
        "responses_stream": {"candidates_total": 1, "selected_total": 1},
    }
    assert snapshot["worker_phase_samples"]["json_parse"] == {
        "samples_total": 1,
        "wall_ns_total": 4_000,
        "cpu_ns_total": 2_000,
        "bytes_total": 512,
        "events_total": 2,
        "wall_us_per_event": 2.0,
        "cpu_us_per_event": 1.0,
    }
    assert snapshot["worker_socket_unread_samples_total"] == 1
    assert snapshot["worker_socket_unread_bytes_total"] == 4096
    assert snapshot["worker_socket_unread_bytes_max"] == 4096
    assert snapshot["worker_socket_unread_bytes_last"] == 4096
    assert snapshot["worker_socket_unread_sample_failures_total"] == 1


def test_threadpool_profile_category_uses_dedicated_names_and_bounded_stack():
    assert worker_runtime._threadpool_category_from_sample(
        "uni-api-json_3",
        ("json/decoder.py:raw_decode:353",),
    ) == ("json_parse", "dedicated_thread_name")
    assert worker_runtime._threadpool_category_from_sample(
        "uni-api-upstream-body_0",
        ("zlib.py:decompress:1",),
    ) == ("upstream_response_decode", "dedicated_thread_name")
    assert worker_runtime._threadpool_category_from_sample(
        "uni-api-idempotency-spool_0",
        ("idempotency_spool.py:_write_all:300",),
    ) == ("idempotency_spool", "dedicated_thread_name")
    assert worker_runtime._threadpool_category_from_sample(
        "uni-api-body_1",
        ("zstandard/backend_c.py:decompress:1",),
    ) == ("request_body_decode", "dedicated_thread_name")
    assert worker_runtime._threadpool_category_from_sample(
        "asyncio_0",
        (
            "concurrent/futures/thread.py:run:58",
            "json/encoder.py:iterencode:258",
        ),
    ) == ("json_serialization", "default_executor_stack")
    assert worker_runtime._threadpool_category_from_sample(
        "MainThread",
        ("json/encoder.py:iterencode:258",),
    ) == (None, None)


def test_sustained_cpu_threshold_triggers_once_then_obeys_cooldown(monkeypatch):
    async def scenario():
        wall = [100.0]
        cpu = [20.0]
        profiles = []
        observer = WorkerRuntimeObserver(
            cpu_profile_enabled=True,
            cpu_profile_trigger_cores=0.9,
            cpu_profile_trigger_samples=2,
            cpu_profile_cooldown_seconds=900,
            monotonic=lambda: wall[0],
            process_time=lambda: cpu[0],
        )

        async def fake_profile(trigger_cpu_cores):
            profiles.append(trigger_cpu_cores)

        monkeypatch.setattr(worker_runtime.sys, "platform", "linux")
        monkeypatch.setattr(observer, "_run_profile", fake_profile)

        wall[0] = 105.0
        cpu[0] = 24.6
        observer.sample_now(emit=False)
        assert profiles == []

        wall[0] = 110.0
        cpu[0] = 29.2
        observer.sample_now(emit=False)
        assert observer._profile_task is not None
        await observer._profile_task
        assert len(profiles) == 1
        assert math.isclose(profiles[0], 0.92)

        wall[0] = 115.0
        cpu[0] = 33.8
        observer.sample_now(emit=False)
        wall[0] = 120.0
        cpu[0] = 38.4
        observer.sample_now(emit=False)
        await asyncio.sleep(0)
        assert len(profiles) == 1

    asyncio.run(scenario())
