import json
import re
import asyncio

import main
import pytest
from fugue_observability import (
    FugueObservabilityClient,
    FugueObservabilityConfig,
    build_uni_api_ember_admission_503_response_write_event,
    build_uni_api_ember_large_body_admission_event,
    build_uni_api_ember_request_telemetry,
    build_uni_api_ember_terminal_hop_metric_event,
    build_uni_api_ember_worker_cpu_profile_event,
    build_uni_api_ember_worker_metric_events,
    build_uni_api_ember_worker_runtime_snapshot_event,
    fugue_observability_config_from_env,
)
from uni_api.admission import (
    Admission503ResponseWriteOutcome,
    RequestAdmissionController,
    RequestBodyObservation,
)


def test_fugue_observability_disabled_without_endpoint(monkeypatch):
    monkeypatch.delenv("FUGUE_OBSERVABILITY_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    config = fugue_observability_config_from_env(service_version="test")

    assert config.enabled is False


def test_worker_metrics_export_rates_cpu_per_mib_and_named_histogram_buckets():
    events = build_uni_api_ember_worker_metric_events(
        service_name="uni-api-ember",
        service_version="1.2.3",
        identity_attrs={"app_id": "app_123", "runtime_id": "runtime_123"},
        snapshot={
            "worker_id": "pod-1:33",
            "worker_cpu_seconds_total": 123.5,
            "worker_cpu_cores": 0.97,
            "worker_single_core_saturation_ratio": 0.97,
            "worker_sse_events_total": 500,
            "worker_sse_bytes_total": 2_000_000,
            "worker_sse_events_per_second": 25.5,
            "worker_sse_bytes_per_second": 500_000.25,
            "worker_inflight_requests": 74,
            "worker_cpu_seconds_per_sse_mebibyte": 1.25,
            "worker_cpu_profile_running": True,
            "worker_cpu_profile_trigger_total": 2,
            "worker_cpu_profile_completed_total": 1,
            "worker_cpu_profile_failed_total": 0,
            "oaix_terminal_flush_to_ember_receive_invalid_total": 0,
            "oaix_terminal_flush_marker_missing_total": 1,
            "oaix_terminal_flush_to_ember_receive_histogram": {
                "count": 3,
                "sum_ms": 8120.5,
                "cumulative_buckets": {"100": 1, "10000": 3},
                "infinite_bucket": 3,
            },
        },
    )
    metrics = {event["metric"]: event for event in events}

    assert metrics["uniapi_ember_worker_cpu_cores"]["value"] == 0.97
    assert metrics["uniapi_ember_worker_inflight_requests"]["value"] == 74
    assert metrics[
        "uniapi_ember_worker_cpu_seconds_per_sse_mebibyte"
    ]["value"] == 1.25
    assert metrics[
        "uniapi_ember_oaix_terminal_flush_to_receive_"
        "milliseconds_bucket_le_10000"
    ]["value"] == 3
    assert all(
        event["attributes"]["component"] == "uni-api-ember-worker"
        for event in events
    )


def test_terminal_hop_raw_metric_has_no_request_id_label():
    event = build_uni_api_ember_terminal_hop_metric_event(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        observation={
            "lag_ms": 7654.25,
            "request_id": "must-not-be-a-label",
            "trace_id": "must-not-be-a-label",
            "terminal_received_at": "2026-07-22T10:00:00+00:00",
        },
    )

    assert event["value"] == 7654.25
    assert event["metric"].endswith("flush_to_receive_milliseconds")
    assert "request_id" not in event["attributes"]
    assert "trace_id" not in event["attributes"]


def test_worker_snapshot_has_queryable_bounded_app_event_fallback():
    event = build_uni_api_ember_worker_runtime_snapshot_event(
        service_name="uni-api-ember",
        service_version="1.7.200",
        identity_attrs={"app_id": "app_123", "runtime_id": "runtime_123"},
        snapshot={
            "worker_metrics_schema_version": 2,
            "worker_id": "pod-1:33",
            "worker_pid": 33,
            "worker_started_at": "2026-07-22T10:00:00+00:00",
            "worker_source_revision": "9114e05",
            "worker_cpu_seconds_total": 123.5,
            "worker_cpu_cores": 0.97,
            "worker_sse_events_per_second": 25.5,
            "worker_sse_bytes_per_second": 500_000.25,
            "worker_inflight_requests": 74,
            "worker_cpu_seconds_per_sse_mebibyte": 1.25,
            "worker_cpu_profile_enabled": True,
            "worker_cpu_profile_running": False,
            "worker_phase_sample_rate": 128,
            "worker_phase_sampling": {
                "responses_stream": {
                    "candidates_total": 256,
                    "selected_total": 2,
                    "secret": "must-not-be-copied",
                },
                "request-controlled-scope": {"selected_total": 999999},
            },
            "worker_phase_samples": {
                "json_parse": {
                    "samples_total": 2,
                    "wall_ns_total": 4000,
                    "cpu_ns_total": 2000,
                    "bytes_total": 512,
                    "events_total": 2,
                    "cpu_us_per_event": 1.0,
                    "request_body": "must-not-be-copied",
                },
                "request-secret-as-phase": {"cpu_ns_total": 999999},
            },
            "worker_threadpool_tasks": {
                "schema_version": 1,
                "lifecycle_semantics": (
                    "explicit_task_tag_wall_thread_cpu_v1"
                ),
                "categories": {
                    "json_parse": {
                        "submitted_total": 2,
                        "completed_total": 2,
                        "failed_total": 0,
                        "queued": 0,
                        "inflight": 0,
                        "cpu_ns_total": 2000,
                        "callback": "must-not-be-copied",
                    },
                    "request-secret-as-category": {
                        "submitted_total": 999999
                    },
                },
                "dedicated_executors": {
                    "json_parse": {
                        "queue_depth": 3,
                        "threads": 2,
                        "alive_threads": 2,
                        "callback": "must-not-be-copied",
                    },
                    "request-controlled-executor": {"queue_depth": 999999},
                },
            },
            "worker_socket_unread_samples_total": 3,
            "worker_socket_unread_bytes_total": 8192,
            "worker_socket_unread_bytes_max": 4096,
            "worker_socket_unread_bytes_last": 1024,
            "worker_socket_unread_sample_failures_total": 1,
            "oaix_terminal_flush_to_ember_receive_histogram": {
                "count": 3,
                "sum_ms": 8120.5,
                "cumulative_buckets": {"100": 1, "10000": 3},
                "infinite_bucket": 3,
            },
            "worker_cpu_profile_latest": {
                "top_stacks": [{"request_body": "must-not-be-copied"}]
            },
            "authorization": "Bearer must-not-be-copied",
        },
    )

    serialized = json.dumps(event, sort_keys=True)
    assert event["event_type"] == "worker_runtime_snapshot"
    assert event["attributes"]["fugue_table"] == "app_events"
    assert event["attributes"]["worker_id"] == "pod-1:33"
    assert event["summary"]["worker_cpu_cores"] == 0.97
    assert event["summary"]["worker_inflight_requests"] == 74
    assert event["summary"]["worker_phase_samples"]["json_parse"][
        "cpu_ns_total"
    ] == 2000
    assert event["summary"]["worker_phase_sampling"]["responses_stream"] == {
        "candidates_total": 256,
        "selected_total": 2,
    }
    assert event["summary"]["worker_threadpool_tasks"]["categories"][
        "json_parse"
    ]["completed_total"] == 2
    assert event["summary"]["worker_threadpool_tasks"][
        "lifecycle_semantics"
    ] == "explicit_task_tag_wall_thread_cpu_v1"
    assert event["summary"]["worker_threadpool_tasks"][
        "dedicated_executors"
    ]["json_parse"] == {
        "queue_depth": 3,
        "threads": 2,
        "alive_threads": 2,
    }
    assert event["summary"]["worker_socket_unread_bytes_max"] == 4096
    assert event["summary"][
        "oaix_terminal_flush_to_ember_receive_histogram"
    ]["cumulative_buckets"]["10000"] == 3
    assert "must-not-be-copied" not in serialized
    assert "request-secret-as-phase" not in serialized
    assert "request-secret-as-category" not in serialized
    assert "request-controlled-scope" not in serialized
    assert "request-controlled-executor" not in serialized
    assert "Bearer" not in serialized


def test_worker_snapshot_posts_durable_event_before_prometheus_metrics():
    async def scenario():
        client = FugueObservabilityClient(
            FugueObservabilityConfig(
                endpoint="https://observability.invalid",
                identity_attrs={"app_id": "app_123"},
            )
        )
        client._queue = asyncio.Queue(maxsize=10)
        calls = []

        async def capture(path, payload):
            calls.append((path, payload))

        client._post_json = capture
        worker = asyncio.create_task(client._worker())
        try:
            assert client.emit_worker_runtime_snapshot(
                {
                    "worker_metrics_schema_version": 1,
                    "worker_id": "pod-1:33",
                    "worker_cpu_cores": 0.97,
                    "worker_inflight_requests": 74,
                }
            )
            await client._queue.join()
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        return calls

    calls = asyncio.run(scenario())
    assert [path for path, _payload in calls] == ["/v1/logs", "/v1/metrics"]
    assert calls[0][1]["events"][0]["event_type"] == (
        "worker_runtime_snapshot"
    )
    assert all(
        event["kind"] == "metric" for event in calls[1][1]["events"]
    )


def test_worker_cpu_profile_event_only_exports_bounded_code_locations():
    event = build_uni_api_ember_worker_cpu_profile_event(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        profile={
            "schema_version": 1,
            "profile_id": "profile-1",
            "worker_id": "pod-1:33",
            "source_revision": "abc123",
            "status": "completed",
            "threadpool_classification_semantics": (
                "explicit_tag_then_dedicated_name_then_bounded_stack_v1"
            ),
            "trigger_cpu_cores": 0.99,
            "started_at": "2026-07-22T10:00:00+00:00",
            "finished_at": "2026-07-22T10:00:10+00:00",
            "sample_rounds": 50,
            "profiled_cpu_seconds": 9.4,
            "top_leaf_functions": [
                {
                    "function": "uni_api/runtime.py:_proxy_responses_stream:7300",
                    "cpu_ticks": 900,
                    "cpu_seconds": 9,
                    "locals": "never-export-request-data",
                }
            ],
            "top_stacks": [
                {
                    "stack": [
                        "uni_api/runtime.py:_stream_worker:6020",
                        "uni_api/runtime.py:_proxy_responses_stream:7300",
                    ],
                    "cpu_ticks": 900,
                    "cpu_seconds": 9,
                    "samples": 45,
                    "request_body": "never-export-request-data",
                }
            ],
            "threadpool_categories": [
                {
                    "category": "json_parse",
                    "cpu_ticks": 750,
                    "cpu_seconds": 7.5,
                    "samples": 38,
                    "sources": [
                        {
                            "source": "dedicated_thread_name",
                            "cpu_ticks": 750,
                            "cpu_seconds": 7.5,
                            "samples": 38,
                            "request_body": "never-export-request-data",
                        },
                        {
                            "source": "request-controlled-source",
                            "cpu_ticks": 999,
                        },
                    ],
                    "request_body": "never-export-request-data",
                },
                {
                    "category": "request-controlled-secret-category",
                    "cpu_ticks": 150,
                    "cpu_seconds": 1.5,
                    "samples": 7,
                },
            ],
            "authorization": "Bearer never-export",
        },
    )

    serialized = json.dumps(event, sort_keys=True)
    assert event["event_type"] == "worker_on_cpu_profile"
    assert event["attributes"]["fugue_table"] == "app_events"
    assert "uni_api/runtime.py:_proxy_responses_stream:7300" in serialized
    assert '"category": "json_parse"' in serialized
    assert '"source": "dedicated_thread_name"' in serialized
    assert "explicit_tag_then_dedicated_name_then_bounded_stack_v1" in serialized
    assert "request-controlled-source" not in serialized
    assert "request-controlled-secret-category" not in serialized
    assert "never-export-request-data" not in serialized
    assert "Bearer never-export" not in serialized


def test_large_body_admission_decision_is_a_body_free_clickhouse_app_event():
    async def scenario():
        decisions = []
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=2,
            wait_timeout_seconds=1,
            max_body_bytes=64,
            body_budget_bytes=128,
            large_body_threshold_weighted_bytes=16,
            large_body_limit=1,
            decision_observer=decisions.append,
        )
        lease = await controller.acquire()
        await lease.reserve_body_bytes(
            17,
            observation=RequestBodyObservation(
                request_id="request-decision",
                trace_id="trace-decision",
                method="POST",
                path="/v1/responses",
                wire_bytes=11,
                decoded_bytes=11,
                json_raw_bytes=11,
                json_structural_item_count=2,
                json_depth=0,
                json_peak_depth=1,
                json_scalar_bytes=0,
                json_estimated_bytes=17,
                json_raw_memory_multiplier=5,
                json_structural_item_memory_bytes=1024,
            ),
        )
        await lease.release()
        return decisions[0]

    decision = asyncio.run(scenario())
    event = build_uni_api_ember_large_body_admission_event(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={
            "tenant_id": "tenant_123",
            "project_id": "project_123",
            "app_id": "app_123",
        },
        decision=decision,
    )

    assert event["event_type"] == "large_body_admission_decision"
    assert event["app_id"] == "app_123"
    assert event["request_id"] == "request-decision"
    assert event["trace_id"] == "trace-decision"
    assert event["attributes"]["fugue_table"] == "app_events"
    assert event["attributes"]["decision"] == "claim"
    assert event["attributes"]["request_self_wire_bytes"] == "11"
    assert event["attributes"]["runtime_global_large_body_active_after"] == "1"
    assert event["summary"]["request_self_json_structural_item_count"] == 2
    assert json.loads(event["summary_json"])["request_self_path"] == "/v1/responses"
    serialized = json.dumps(event, sort_keys=True)
    assert "authorization" not in serialized.lower()
    assert "request body content" not in serialized.lower()


def test_admission_503_write_outcome_is_a_joinable_body_free_app_event():
    event = build_uni_api_ember_admission_503_response_write_event(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        outcome=Admission503ResponseWriteOutcome(
            schema_version=1,
            occurred_at_unix_ms=1_000_000,
            reason="large_body_capacity_exhausted",
            intended_status_code=503,
            asgi_response_write_completed=False,
            request_self_lease_id="lease-safe-1",
            request_self_request_id="request-safe-1",
            request_self_trace_id="trace-safe-1",
            request_self_method="POST",
            request_self_path="/v1/responses",
            runtime_global_admission_503_response_write_completed_total_after=6,
            runtime_global_admission_503_response_write_failed_total_after=1,
        ),
    )

    assert event["event_type"] == "admission_503_response_write_outcome"
    assert event["attributes"]["fugue_table"] == "app_events"
    assert event["request_id"] == "request-safe-1"
    assert event["trace_id"] == "trace-safe-1"
    assert event["status_code"] == 503
    assert event["summary"]["asgi_response_write_completed"] is False
    assert json.loads(event["summary_json"])["request_self_lease_id"] == (
        "lease-safe-1"
    )


def test_large_body_decision_queue_drop_is_visible_to_controller_and_exporter():
    async def scenario():
        client = FugueObservabilityClient(
            FugueObservabilityConfig(endpoint="https://observability.invalid")
        )
        client._queue = asyncio.Queue(maxsize=1)
        client._queue.put_nowait(("/v1/logs", {"events": [{}]}))
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=64,
            body_budget_bytes=64,
            large_body_threshold_weighted_bytes=16,
            large_body_limit=1,
            decision_observer=client.emit_large_body_admission_decision,
        )
        lease = await controller.acquire()
        await lease.reserve_body_bytes(17)
        snapshot = controller.snapshot()
        delivery = client.delivery_snapshot()
        await lease.release()
        return snapshot, delivery

    snapshot, delivery = asyncio.run(scenario())
    assert snapshot["large_body_decision_observer_enqueue_failures_total"] == 1
    assert delivery["large_body_decision_enqueue_dropped_total"] == 1


def test_body_complexity_diagnostics_reach_fact_log_and_dedicated_span_safely():
    sentinel = "never-export-this-request-body"
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        current_info={
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "request_id": "request_body_complexity_123",
            "endpoint": "POST /v1/responses",
            "status_code": 413,
            "process_time": 0.004,
            "body": sentinel,
            "request_body_complexity": {
                "schema_version": 1,
                "reason": "max_estimated_bytes",
                "trigger_phase": "structural_item_scan",
                "raw_bytes": 2048,
                "structural_item_count": 8192,
                "depth": 7,
                "peak_depth": 14,
                "scalar_bytes": 0,
                "estimated_bytes": 9_000_000,
                "configured_limit": 8_000_000,
                "max_depth": 128,
                "max_scalar_bytes": 4096,
                "max_estimated_bytes": 8_000_000,
                "raw_memory_multiplier": 5,
                "structural_item_memory_bytes": 1024,
                "reserved_weighted_bytes_at_rejection": 7_000_000,
                "json_memory_reserved_target_bytes_at_rejection": 6_000_000,
                "body": sentinel,
                "authorization": "Bearer never-export",
            },
            "timing_spans": {
                "request_received": 0,
                "request_body_rejected": 3,
                "stream_end": 4,
            },
        },
        runtime_metrics={},
    )

    summary_log = telemetry["logs"][0]
    expected = {
        "request_body_complexity_schema_version": "1",
        "request_body_complexity_reason": "max_estimated_bytes",
        "request_body_complexity_trigger_phase": "structural_item_scan",
        "request_body_complexity_raw_bytes": "2048",
        "request_body_complexity_structural_item_count": "8192",
        "request_body_complexity_depth": "7",
        "request_body_complexity_peak_depth": "14",
        "request_body_complexity_scalar_bytes": "0",
        "request_body_complexity_estimated_bytes": "9000000",
        "request_body_complexity_configured_limit": "8000000",
        "request_body_complexity_max_depth": "128",
        "request_body_complexity_max_scalar_bytes": "4096",
        "request_body_complexity_max_estimated_bytes": "8000000",
        "request_body_complexity_raw_memory_multiplier": "5",
        "request_body_complexity_structural_item_memory_bytes": "1024",
        "request_body_reserved_weighted_bytes_at_rejection": "7000000",
        "json_memory_reserved_target_bytes_at_rejection": "6000000",
    }
    for key, value in expected.items():
        assert summary_log["summary"][key] == value
        assert summary_log["attributes"][key] == value

    rejection_spans = [
        event
        for event in telemetry["traces"]
        if event["attributes"].get("stage") == "request_body_rejected"
    ]
    assert len(rejection_spans) == 1
    for key, value in expected.items():
        assert rejection_spans[0]["attributes"][key] == value

    serialized = json.dumps(telemetry, sort_keys=True)
    assert sentinel not in serialized
    assert "Bearer never-export" not in serialized
    assert "request_body_json_tokens" not in serialized
    for metric in telemetry["metrics"]:
        assert not any(
            key.startswith("request_body_complexity_")
            for key in metric["attributes"]
        )


def test_normal_request_does_not_emit_body_complexity_diagnostics():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        current_info={
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "request_id": "normal_request_123",
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "process_time": 0.004,
            "timing_spans": {"request_received": 0, "stream_end": 4},
        },
        runtime_metrics={},
    )

    serialized = json.dumps(telemetry, sort_keys=True)
    assert "request_body_complexity_" not in serialized
    assert "request_body_rejected" not in serialized


def test_transport_failure_classification_reaches_logs_summary_and_metrics():
    failure_facts = {
        "transport_error_kind": "pool_timeout",
        "transport_error_owner": "ember_local_overload",
        "transport_error_phase": "pool_acquire",
        "transport_error_status_code": 503,
        "provider_penalty_eligible": False,
        "local_overload": True,
        "transport_dns_duration_ms": 12,
        "transport_tcp_duration_ms": 80,
        "transport_first_body_ms": 102,
    }
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        current_info={
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "request_id": "pool_timeout_123",
            "endpoint": "POST /v1/responses",
            "status_code": 503,
            "process_time": 0.25,
            "attempt_count": 1,
            "transport_error_count": 1,
            "local_overload_count": 1,
            "timing_spans": {"request_received": 0, "stream_end": 250},
            "routing_attempts": [
                {
                    "index": 1,
                    "provider": "provider-a",
                    "actual_model": "model-a",
                    "semantic_status_code": 503,
                    "error_type": "PoolTimeout",
                    **failure_facts,
                }
            ],
            "upstream_attempts": [
                {
                    "index": 1,
                    "provider": "provider-a",
                    "actual_model": "model-a",
                    "status_code": 503,
                    "error_type": "PoolTimeout",
                    **failure_facts,
                }
            ],
        },
        runtime_metrics={},
    )

    request_summary = next(
        event
        for event in telemetry["logs"]
        if event["event"] == "request_summary"
    )
    assert request_summary["summary"]["transport_error_count"] == "1"
    assert request_summary["summary"]["local_overload_count"] == "1"
    assert request_summary["summary"]["transport_dns_duration_ms"] == "12"
    assert request_summary["summary"]["transport_tcp_duration_ms"] == "80"
    assert request_summary["summary"]["transport_first_body_ms"] == "102"


    for event_type in ("routing_attempt", "upstream_attempt"):
        event = next(
            item
            for item in telemetry["logs"]
            if item["event"] == event_type
        )
        attrs = event["attributes"]
        assert attrs["transport_error_kind"] == "pool_timeout"
        assert attrs["transport_error_owner"] == "ember_local_overload"
        assert attrs["transport_error_phase"] == "pool_acquire"
        assert attrs["transport_error_status_code"] == "503"
        assert attrs["provider_penalty_eligible"] == "false"
        assert attrs["local_overload"] == "true"
        assert attrs["transport_dns_duration_ms"] == "12"
        assert attrs["transport_tcp_duration_ms"] == "80"
        assert attrs["transport_first_body_ms"] == "102"

    metrics = {event["metric"]: event for event in telemetry["metrics"]}
    assert metrics["uniapi_ember_transport_errors_total"]["value"] == 1
    assert metrics["uniapi_ember_local_overload_total"]["value"] == 1


def test_request_summary_exports_bounded_rust_responses_fields():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "request_kind": "/v1/responses",
            "status_code": 200,
            "stream": True,
            "rust_responses_data_plane": True,
            "rust_responses_control_version": 1,
            "rust_responses_external_committed": True,
            "rust_responses_commit_reason": "real_output",
            "rust_responses_precommit_events": 2,
            "rust_responses_precommit_bytes": 120,
            "rust_responses_upstream_bytes": 900,
            "rust_responses_downstream_bytes": 700,
            "rust_responses_sse_events": 8,
            "rust_responses_delta_events": 4,
            "rust_responses_normalized_events": 3,
            "rust_request_spool": True,
            "rust_request_spool_body_bytes": 20_000_000,
            "rust_request_spool_memory_peak_bytes": 65_536,
            "rust_request_spool_local_disk_bytes": 20_000_000,
            "rust_request_spool_local_free_bytes_start": 80_000_000_000,
            "rust_request_spool_local_writable_bytes_start": 70_000_000_000,
            "rust_request_spool_local_free_inodes_start": 8_000_000,
            "rust_request_spool_local_writable_inodes_start": 7_500_000,
            "rust_request_spool_resource_wait_ms": 17,
            "rust_request_spool_final_tier": "local_disk",
            "rust_request_spool_failure_resource": "memory_headroom",
        },
        runtime_metrics={},
    )
    request_summary = next(
        row
        for row in telemetry["logs"]
        if row.get("event") == "request_summary"
    )
    expected = {
        "rust_responses_data_plane": "true",
        "rust_responses_control_version": "1",
        "rust_responses_external_committed": "true",
        "rust_responses_commit_reason": "real_output",
        "rust_responses_precommit_events": "2",
        "rust_responses_precommit_bytes": "120",
        "rust_responses_upstream_bytes": "900",
        "rust_responses_downstream_bytes": "700",
        "rust_responses_sse_events": "8",
        "rust_responses_delta_events": "4",
        "rust_responses_normalized_events": "3",
        "rust_request_spool": "true",
        "rust_request_spool_body_bytes": "20000000",
        "rust_request_spool_memory_peak_bytes": "65536",
        "rust_request_spool_local_disk_bytes": "20000000",
        "rust_request_spool_local_free_bytes_start": "80000000000",
        "rust_request_spool_local_writable_bytes_start": "70000000000",
        "rust_request_spool_local_free_inodes_start": "8000000",
        "rust_request_spool_local_writable_inodes_start": "7500000",
        "rust_request_spool_resource_wait_ms": "17",
        "rust_request_spool_final_tier": "local_disk",
        "rust_request_spool_failure_resource": "memory_headroom",
    }
    assert {
        key: request_summary["summary"].get(key) for key in expected
    } == expected


def test_early_admission_rejection_only_emits_observed_stage_spans():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        current_info={
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "request_id": "early_admission_rejection_123",
            "endpoint": "POST /v1/responses",
            "status_code": 503,
            "admission_rejected": True,
            "error_type": "large_body_capacity_exhausted",
            "process_time": 0.01,
            "timing_spans": {"request_received": 0},
        },
        runtime_metrics={},
    )

    assert [
        (event["attributes"]["stage"], event["attributes"]["stage_ms"])
        for event in telemetry["traces"]
    ] == [("request_received", "0")]


def test_client_pool_stage_is_emitted_from_observed_pool_timing():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        current_info={
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "request_id": "client_pool_timing_123",
            "endpoint": "POST /v1/responses",
            "status_code": 503,
            "process_time": 0.01,
            "timing_spans": {
                "request_received": 0,
                "client_pool_acquire_start": 2,
                "client_pool_acquire_end": 7,
                "client_pool_acquired": 7,
                "upstream_pool_wait_ms": 5,
            },
        },
        runtime_metrics={},
    )

    assert [
        event["attributes"]["stage"] for event in telemetry["traces"]
    ] == ["request_received", "client_pool_acquired"]
    client_pool_span = telemetry["traces"][1]["attributes"]
    assert client_pool_span["stage_ms"] == "5"
    assert client_pool_span["client_pool_acquire_start_ms"] == "2"
    assert client_pool_span["client_pool_acquire_end_ms"] == "7"


def test_failed_client_pool_attempt_does_not_emit_acquired_span():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        current_info={
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "request_id": "client_pool_rejected_123",
            "endpoint": "POST /v1/responses",
            "status_code": 503,
            "process_time": 0.01,
            "timing_spans": {
                "request_received": 0,
                "client_pool_acquire_start": 2,
                "client_pool_acquire_end": 7,
                "upstream_pool_wait_ms": 5,
            },
        },
        runtime_metrics={},
    )

    assert [
        event["attributes"]["stage"] for event in telemetry["traces"]
    ] == ["request_received"]


def test_uni_api_ember_telemetry_redacts_secrets_and_body():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"tenant_id": "tenant_123", "app_id": "app_123"},
        current_info={
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "request_id": "request_123",
            "parent_span_id": "00f067aa0ba902b7",
            "endpoint": "POST /v1/responses",
            "model": "gpt-5.4",
            "provider": "oaix",
            "role": "sk-test",
            "stream": True,
            "status_code": 200,
            "wire_status_code": 200,
            "response_committed": True,
            "process_time": 1.25,
            "api_key": "sk-secret-api-key",
            "text": "this is request body content",
            "authorization": "Bearer ember-secret-token",
            "headers": {
                "Authorization": "Bearer ember-secret-token",
                "Cookie": "session=ember-cookie-secret",
            },
            "cookie": "session=ember-cookie-secret",
            "database_url": "postgresql://user:pass@db/ember",
            "body": {"input": "ember request body secret"},
            "email": "ember@example.com",
            "source_ip": "203.0.113.88",
            "token": "ember-upstream-token-secret",
            "message_roles": "system/user",
            "role_counts": "system:1,user:1",
            "retry_count": 1,
            "attempt_count": 2,
            "retry_decision_count": 1,
            "retry_transition_count": 1,
            "planned_attempt_count": 12,
            "planned_retry_count": 10,
            "matching_provider_count": 2,
            "routing_attempts_omitted_count": 0,
            "cooldown_count": 1,
            "timing_spans": {
                "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
                "parent_span_id": "00f067aa0ba902b7",
                "request_received": 0,
                "body_parsed": 3,
                "provider_selected": 8,
                "provider_key_selected": 10,
                "retry_started": 20,
                "retry_count": 1,
                "retry_status_code": 503,
                "retry_provider": "oaix",
                "upstream_pool_wait_ms": 17,
                "client_pool_acquired": 27,
                "Authorization": "Bearer ember-secret-token",
                "Cookie": "session=ember-cookie-secret",
                "database_url": "postgresql://user:pass@db/ember",
                "body": "ember request body secret",
                "email": "ember@example.com",
                "source_ip": "203.0.113.88",
                "upstream_send_start": 30,
                "upstream_headers_received": 90,
                "upstream_first_chunk": 140,
                "downstream_response_start": 145,
                "stream_end": 1250,
            },
            "upstream_attempts": [
                {
                    "index": 1,
                    "endpoint": "/v1/responses/compact",
                    "provider": "fugue-codex",
                    "model": "gpt-5.4",
                    "actual_model": "gpt-5.4",
                    "engine": "codex",
                    "upstream_host": "oaix.internal",
                    "payload_bytes": 29658219,
                    "timeout_seconds": 120,
                    "timeout_adjusted_from_seconds": 20,
                    "wants_compact": True,
                    "stream": False,
                    "started_ms": 31,
                    "duration_ms": 20061,
                    "status_code": 504,
                    "success": False,
                    "error_type": "ReadTimeout",
                    "error_message": "Bearer ember-secret-token",
                    "stream_diagnostics": {
                        "semantic_status": "error",
                        "diagnosis": "responses_partial_event_abort",
                        "failure_stage": "postcommit",
                        "oaix_connection_id": "oaixc-observe-1",
                        "http_version": "HTTP/2",
                        "httpcore_stream_id": 17,
                        "explicit_proxy_configured": False,
                        "transport_local_endpoint_hmac": "a" * 64,
                        "transport_peer_endpoint_hmac": "b" * 64,
                        "transport_four_tuple_hmac": "c" * 64,
                        "transport_socket_hmac": "d" * 64,
                        "upstream_body_bytes": 4096,
                        "upstream_chunk_count": 8,
                        "complete_event_count": 5,
                        "last_event_type": "response.output_text.delta",
                        "last_event_ordinal": 5,
                        "last_event_bytes": 128,
                        "last_event_sha256": "e" * 64,
                        "partial_event_bytes": 77,
                        "partial_event_sha256": "f" * 64,
                        "hash_scope": "ember_normalized_sse_event_lf_v1",
                        "partial_hash_scope": "normalized_prefix_plus_utf8_tail_v1",
                        "upstream_eof_seen": False,
                        "upstream_terminal_seen": False,
                        "upstream_terminal_validated": False,
                        "downstream_terminal_seen": False,
                        "downstream_terminal_asgi_write_completed": False,
                        "error_event_seen": True,
                        "usage_seen": False,
                        "exception_type": "ReadError",
                        "exception_origin": "postcommit_stream",
                        "exception_errno": 104,
                        "exception_errno_name": "ECONNRESET",
                        "exception_chain_depth": 3,
                        "exception_chain_truncated": False,
                        "exception_chain": [
                            {"type": "ReadError", "relation": "raised"},
                            {"type": "ReadError", "relation": "cause"},
                            {
                                "type": "ConnectionResetError",
                                "relation": "cause",
                                "errno": 104,
                            },
                        ],
                        "cleanup_owner": "responses_proxy_finally",
                        "cleanup_trigger": "after_upstream_read_or_stream_failure",
                        "cleanup_method": "cooperative_response_aclose",
                        "cleanup_result": "succeeded",
                        "cleanup_transport_evicted": False,
                        "cleanup_transport_safe": True,
                    },
                }
            ],
            "routing_attempts": [
                {
                    "index": 1,
                    "provider": "fugue-codex",
                    "model": "gpt-5.4",
                    "actual_model": "gpt-5.4",
                    "started_ms": 30,
                    "duration_ms": 61,
                    "semantic_status_code": 503,
                    "outcome": "retry_decided",
                    "success": False,
                    "error_type": "ReadTimeout",
                    "error_message_sha256": "9" * 64,
                    "error_message": "Bearer ember-secret-token",
                    "retry_decision": True,
                    "retry_reason": "http_503:ReadTimeout",
                    "retry_transition_to_index": 2,
                },
                {
                    "index": 2,
                    "provider": "oaix",
                    "model": "gpt-5.4",
                    "actual_model": "gpt-5.4",
                    "started_ms": 92,
                    "duration_ms": 1158,
                    "wire_status_code": 200,
                    "semantic_status_code": 400,
                    "terminal_event_type": "error",
                    "outcome": "semantic_failure_terminal",
                    "success": False,
                    "error_code": "context_length_exceeded",
                    "error_type": "invalid_request_error",
                    "error_message_sha256": "8" * 64,
                    "retry_decision": False,
                },
            ],
            "responses_stream_diagnostics": {
                "semantic_status": "error",
                "diagnosis": "responses_partial_event_abort",
                "failure_stage": "postcommit",
                "oaix_connection_id": "oaixc-observe-1",
                "upstream_body_bytes": 4096,
                "upstream_chunk_count": 8,
                "last_event_type": "response.output_text.delta",
                "last_event_ordinal": 5,
                "last_event_bytes": 128,
                "last_event_sha256": "e" * 64,
                "partial_event_bytes": 77,
                "partial_event_sha256": "f" * 64,
                "hash_scope": "ember_normalized_sse_event_lf_v1",
                "upstream_terminal_seen": False,
                "upstream_terminal_validated": False,
                "downstream_terminal_seen": False,
                "downstream_terminal_asgi_write_completed": False,
                "error_event_seen": True,
                "usage_seen": False,
                "exception_type": "ReadError",
                "exception_origin": "postcommit_stream",
                "cleanup_result": "succeeded",
            },
        },
        runtime_metrics={
            "inflight_requests": 12,
            "request_admission_mode": "weighted_resources",
            "request_control_memory_reserved_bytes": 786432,
            "cpu_phase_capacity": 1,
            "cpu_phase_active": 1,
            "cpu_phase_waiters": 3,
            "request_body_reserved_weighted_bytes": 8192,
            "upstream_response_reserved_weighted_bytes": 16384,
            "request_retained_reserved_weighted_bytes": 24576,
            "runtime_global_request_body_reserved_weighted_bytes": 8192,
            "runtime_global_upstream_response_reserved_weighted_bytes": 16384,
            "runtime_global_retained_reserved_weighted_bytes": 24576,
            "runtime_global_large_body_active": 1,
            "runtime_global_admission_rejection_decision_total": 7,
            "runtime_global_admission_503_response_write_completed_total": 6,
            "runtime_global_admission_503_response_write_failed_total": 1,
            "request_deferred_memory_requests": 2,
            "request_deferred_memory_weighted_bytes": 4096,
            "runtime_global_deferred_memory_requests": 2,
            "runtime_global_deferred_memory_weighted_bytes": 4096,
            "waiting_first_byte": 4,
            "event_loop_lag_ms": 2,
            "upstream_pool_in_use": 3,
            "outbound_network_resources": {
                "fd_headroom": 900000,
                "ephemeral_port_headroom": 11000,
            },
            "stream_parser_reserved_bytes": 2048,
            "stream_parser_rejected_total": 1,
        },
    )

    serialized = json.dumps(telemetry, sort_keys=True)
    assert "sk-secret-api-key" not in serialized
    assert "this is request body content" not in serialized
    for secret in {
        "Bearer ember-secret-token",
        "ember-cookie-secret",
        "postgresql://user:pass@db/ember",
        "ember request body secret",
        "ember@example.com",
        "203.0.113.88",
        "ember-upstream-token-secret",
    }:
        assert secret not in serialized
    assert "api_key_hash" in serialized
    assert "system/user" in serialized

    log_event = telemetry["logs"][0]
    assert log_event["level"] == "info"
    assert log_event["service"] == "uni-api-ember"
    assert log_event["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert log_event["request_id"] == "request_123"
    assert log_event["event"] == "request_summary"
    assert log_event["event_type"] == "request_summary"
    assert log_event["message"] == "uni-api-ember request finished"
    assert log_event["app_id"] == "app_123"
    assert log_event["path"] == "/v1/responses"
    assert log_event["status_code"] == 200
    summary = log_event["summary"]
    assert summary["attempt_count"] == "2"
    assert summary["retry_decision_count"] == "1"
    assert summary["retry_transition_count"] == "1"
    assert summary["planned_attempt_count"] == "12"
    assert summary["wire_status_code"] == "200"
    assert summary["semantic_status"] == "error"
    assert summary["upstream_terminal_seen"] == "false"
    assert summary["upstream_terminal_validated"] == "false"
    assert summary["downstream_terminal_seen"] == "false"
    assert summary["usage_seen"] == "false"
    assert summary["diagnosis"] == "responses_partial_event_abort"
    assert summary["request_admission_mode"] == "weighted_resources"
    assert summary["request_control_memory_reserved_bytes"] == "786432"
    assert summary["outbound_fd_headroom"] == "900000"
    assert summary["outbound_ephemeral_port_headroom"] == "11000"
    assert summary["failure_stage"] == "postcommit"
    assert summary["oaix_connection_id"] == "oaixc-observe-1"
    attempt_event = next(event for event in telemetry["logs"] if event["event"] == "upstream_attempt")
    attempt_attrs = attempt_event["attributes"]
    assert attempt_attrs["provider"] == "fugue-codex"
    assert attempt_attrs["attempt_status_code"] == "504"
    assert attempt_attrs["attempt_error_type"] == "ReadTimeout"
    assert attempt_attrs["payload_bytes"] == "29658219"
    assert attempt_attrs["timeout_seconds"] == "120"
    assert attempt_attrs["timeout_adjusted_from_seconds"] == "20"
    assert attempt_attrs["diagnosis"] == "responses_partial_event_abort"
    assert attempt_attrs["failure_stage"] == "postcommit"
    assert attempt_attrs["upstream_terminal_validated"] == "false"
    assert attempt_attrs["oaix_connection_id"] == "oaixc-observe-1"
    assert attempt_attrs["upstream_http_version"] == "HTTP/2"
    assert attempt_attrs["exception_errno_name"] == "ECONNRESET"
    assert attempt_attrs["exception_chain_depth"] == "3"
    assert "ConnectionResetError" in attempt_attrs["exception_chain_json"]
    assert attempt_attrs["cleanup_owner"] == "responses_proxy_finally"
    routing_events = [
        event for event in telemetry["logs"] if event["event"] == "routing_attempt"
    ]
    assert len(routing_events) == 2
    first_routing = routing_events[0]["attributes"]
    assert first_routing["attempt_index"] == "1"
    assert first_routing["semantic_status_code"] == "503"
    assert first_routing["retry_decision"] == "true"
    assert first_routing["retry_transition_to_index"] == "2"
    assert first_routing["error_message_sha256"] == "9" * 64
    second_routing = routing_events[1]["attributes"]
    assert second_routing["wire_status_code"] == "200"
    assert second_routing["semantic_status_code"] == "400"
    assert second_routing["terminal_event_type"] == "error"
    assert second_routing["attempt_error_code"] == "context_length_exceeded"

    stages = {
        event["attributes"]["stage"]
        for event in telemetry["traces"]
    }
    stage_ms = {
        event["attributes"]["stage"]: event["attributes"]["stage_ms"]
        for event in telemetry["traces"]
    }
    assert {
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
    }.issubset(stages)
    assert stage_ms["upstream_first_chunk"] == "140"
    assert stage_ms["downstream_response_start"] == "5"

    for metric in telemetry["metrics"]:
        attrs = metric["attributes"]
        assert "trace_id" not in attrs
        assert "request_id" not in attrs
        assert "api_key_hash" not in attrs

    metrics = {event["metric"]: event for event in telemetry["metrics"]}
    assert metrics["uniapi_ember_upstream_errors_total"]["value"] == 1
    assert metrics["uniapi_ember_attempt_total"]["value"] == 2
    assert metrics["uniapi_ember_cpu_phase_active"]["value"] == 1
    assert metrics["uniapi_ember_cpu_phase_waiters"]["value"] == 3
    assert metrics[
        "uniapi_ember_request_control_memory_reserved_bytes"
    ]["value"] == 786432
    assert metrics["uniapi_ember_outbound_fd_headroom"]["value"] == 900000
    assert metrics[
        "uniapi_ember_outbound_ephemeral_port_headroom"
    ]["value"] == 11000
    assert metrics["uniapi_ember_retry_decision_total"]["value"] == 1
    assert metrics["uniapi_ember_retry_transition_total"]["value"] == 1
    assert metrics["uniapi_ember_retry_total"]["value"] == 1
    assert metrics["uniapi_ember_exposed_5xx_total"]["value"] == 0
    assert (
        metrics["uniapi_ember_request_body_reserved_weighted_bytes"]["value"]
        == 8192
    )
    assert metrics["uniapi_ember_request_body_reserved_weighted_bytes"][
        "attributes"
    ]["metric_scope"] == "runtime_global"
    assert metrics["uniapi_ember_request_body_reserved_weighted_bytes"][
        "attributes"
    ]["legacy_alias_of"] == (
        "uniapi_ember_runtime_global_request_body_reserved_weighted_bytes"
    )
    assert (
        metrics[
            "uniapi_ember_runtime_global_request_body_reserved_weighted_bytes"
        ]["value"]
        == 8192
    )
    assert (
        metrics[
            "uniapi_ember_upstream_response_reserved_weighted_bytes"
        ]["value"]
        == 16384
    )
    assert (
        metrics[
            "uniapi_ember_request_retained_reserved_weighted_bytes"
        ]["value"]
        == 24576
    )
    assert metrics["uniapi_ember_request_deferred_memory_requests"]["value"] == 2
    assert metrics["uniapi_ember_runtime_global_deferred_memory_requests"][
        "value"
    ] == 2
    assert (
        metrics["uniapi_ember_request_deferred_memory_weighted_bytes"]["value"]
        == 4096
    )
    assert metrics["uniapi_ember_stream_parser_reserved_bytes"]["value"] == 2048
    assert metrics["uniapi_ember_stream_parser_rejected_total"]["value"] == 1
    assert metrics[
        "uniapi_ember_runtime_global_admission_rejection_decision_total"
    ]["value"] == 7
    assert metrics[
        "uniapi_ember_runtime_global_admission_503_response_write_completed_total"
    ]["value"] == 6
    assert metrics[
        "uniapi_ember_runtime_global_admission_503_response_write_failed_total"
    ]["value"] == 1
    assert "route_id" not in metrics["uniapi_ember_inflight_requests"]["attributes"]
    assert metrics["uniapi_ember_inflight_requests"]["attributes"][
        "metric_scope"
    ] == "runtime_global"
    assert metrics["uniapi_ember_request_duration_ms"]["attributes"]["route_id"]
    assert metrics["uniapi_ember_request_duration_ms"]["attributes"][
        "metric_scope"
    ] == "request_self"


def test_local_admission_503_is_not_counted_as_upstream_failure():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 503,
            "admission_rejected": True,
            "error_type": "queue_full",
            "process_time": 0.01,
            "upstream_attempts": [],
        },
        runtime_metrics={"inflight_requests": 100, "request_waiters": 900},
    )

    metrics = {event["metric"]: event["value"] for event in telemetry["metrics"]}
    assert metrics["uniapi_ember_upstream_errors_total"] == 0
    assert metrics["uniapi_ember_exposed_5xx_total"] == 1
    assert metrics["uniapi_ember_request_admission_rejected_total"] == 1


def test_post_commit_stream_failure_keeps_wire_200_and_has_failure_metric():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "wire_status_code": 200,
            "stream": True,
            "stream_outcome": "local_backpressure_abort",
            "stream_error_status_code": 503,
            "stream_error_after_response_start": True,
            "error_type": "StreamQueuePutTimeout",
            "process_time": 1.0,
        },
    )

    log_event = telemetry["logs"][0]
    assert log_event["status_code"] == 200
    assert log_event["level"] == "error"
    assert log_event["attributes"]["stream_outcome"] == "local_backpressure_abort"
    assert log_event["attributes"]["stream_error_status_code"] == "503"
    metrics = {event["metric"]: event["value"] for event in telemetry["metrics"]}
    assert metrics["uniapi_ember_exposed_5xx_total"] == 0
    assert metrics["uniapi_ember_stream_failures_total"] == 1


def test_responses_diagnostics_are_exported_with_valid_bounded_json():
    diagnostic = {
        "schema_version": 1,
        "semantic_status": "error",
        "diagnosis": "responses_read_error",
        "failure_stage": "upstream_headers",
        "terminal_consistency_status": "inconsistent",
        "terminal_semantics_consistent": False,
        "terminal_semantics_inconsistency": ["declared_outcome_mismatch"],
        "usage_object_seen": True,
        "usage_counters_seen": True,
        "usage_input_known": True,
        "usage_output_known": False,
        "usage_total_known": True,
        "usage_values_valid": True,
        "usage_seen": False,
        "downstream_usage_object_seen": True,
        "downstream_usage_counters_seen": True,
        "downstream_usage_input_known": True,
        "downstream_usage_output_known": False,
        "downstream_usage_total_known": True,
        "downstream_usage_values_valid": True,
        "downstream_usage_seen": False,
        "downstream_usage_observer_status": "completed",
        "transport_error_code": "peer_closed_incomplete_chunked_body",
        "transport_error_code_source": "known_message_pattern",
        "transport_end_trigger": "httpcore_body_read_failure",
        "response_start_asgi_write_attempted": True,
        "response_start_asgi_write_completed": True,
        "downstream_final_body_attempted": False,
        "downstream_final_body_completed": False,
        "cleanup_result": "incomplete",
        "cleanup_failure": True,
        "cleanup_failure_stage": "cleanup",
        "cleanup_actions": [
            {
                "actor": "responses_proxy_finally",
                "method": "cooperative_response_aclose",
                "transport_safe": False,
                "padding": "x" * 300,
            }
            for _ in range(32)
        ],
        "cleanup_actions_truncated": True,
        "terminal_timeline_schema_version": 1,
        "terminal_timeline_clock": "unix_nano_plus_process_monotonic_v1",
        "terminal_received_semantics": "upstream_iterator_yield_completed_v1",
        "terminal_timeline_complete": True,
        "terminal_received_unix_nano": 1_700_000_000_000_000_000,
        "terminal_received_monotonic_nano": 10_000,
        "terminal_parse_completed_unix_nano": 1_700_000_000_000_001_000,
        "terminal_parse_completed_monotonic_nano": 11_000,
        "terminal_received_from_receive_us": 0.0,
        "terminal_parse_completed_from_receive_us": 1.0,
        "terminal_received_to_parse_completed_us": 1.0,
        "phase_sample_rate": 128,
        "phase_sampled": True,
        "phase_cpu_semantics": "calling_thread_elapsed_inclusive_v1",
        "phase_bytes_semantics": "known_wire_bytes_only_v1",
        "phase_json_parse_samples": 4,
        "phase_json_parse_wall_ns": 4000,
        "phase_json_parse_cpu_ns": 2000,
        "phase_json_parse_bytes": 512,
        "phase_json_parse_events": 4,
        "socket_unread_sample_rate": 128,
        "socket_unread_semantics": (
            "linux_connection_receive_queue_after_chunk_v1"
        ),
        "socket_unread_samples": 2,
        "socket_unread_bytes_total": 8192,
        "socket_unread_bytes_max": 4096,
        "socket_unread_bytes_last": 4096,
        "socket_unread_sample_failures": 0,
        "exception_chain": [
            {
                "relation": "cause",
                "type": "RemoteProtocolError",
                "message_sha256": "a" * 64,
                "padding": "y" * 300,
            }
            for _ in range(32)
        ],
        "exception_chain_truncated": True,
    }
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "wire_status_code": 200,
            "response_committed": True,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "usage_parse_error": "missing_usage_components",
            "responses_stream_diagnostics": diagnostic,
            "upstream_attempts": [
                {
                    "index": 1,
                    "status_code": 200,
                    "stream_diagnostics": diagnostic,
                }
            ],
        },
    )

    summary = telemetry["logs"][0]["summary"]
    attempt = telemetry["logs"][1]["attributes"]
    for attrs in (summary, attempt):
        assert attrs["failure_stage"] == "precommit"
        assert attrs["transport_error_code"] == (
            "peer_closed_incomplete_chunked_body"
        )
        assert attrs["terminal_semantics_consistent"] == "false"
        assert attrs["usage_input_known"] == "true"
        assert attrs["usage_output_known"] == "false"
        assert attrs["downstream_usage_output_known"] == "false"
        assert attrs["cleanup_failure_stage"] == "cleanup"
        assert attrs["terminal_timeline_complete"] == "true"
        assert attrs["terminal_received_unix_nano"] == (
            "1700000000000000000"
        )
        assert attrs["terminal_received_to_parse_completed_us"] == "1"
        assert attrs["phase_sampled"] == "true"
        assert attrs["phase_json_parse_cpu_ns"] == "2000"
        assert attrs["socket_unread_bytes_max"] == "4096"
        for field in (
            "terminal_semantics_inconsistency_json",
            "exception_chain_json",
            "cleanup_actions_json",
        ):
            json.loads(attrs[field])
            assert len(attrs[field].encode("utf-8")) <= 4096
        assert attrs["exception_chain_json_truncated"] == "true"
        assert attrs["cleanup_actions_json_truncated"] == "true"
        assert len(attrs["exception_chain_json_sha256"]) == 64

    assert summary["response_committed"] == "true"
    assert summary["prompt_tokens"] == "0"
    assert "completion_tokens" not in summary
    assert summary["total_tokens"] == "0"
    assert summary["usage_parse_error"] == "missing_usage_components"


def test_responses_summary_exports_real_known_zero_usage():
    no_diagnostics = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )
    no_diagnostics_summary = no_diagnostics["logs"][0]["summary"]
    assert "prompt_tokens" not in no_diagnostics_summary
    assert "completion_tokens" not in no_diagnostics_summary
    assert "total_tokens" not in no_diagnostics_summary

    unknown = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "responses_stream_diagnostics": {
                "downstream_usage_input_known": False,
                "downstream_usage_output_known": False,
                "downstream_usage_total_known": False,
                "downstream_usage_seen": False,
            },
        },
    )
    unknown_summary = unknown["logs"][0]["summary"]
    assert "prompt_tokens" not in unknown_summary
    assert "completion_tokens" not in unknown_summary
    assert "total_tokens" not in unknown_summary

    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "responses_stream_diagnostics": {
                "downstream_usage_input_known": True,
                "downstream_usage_output_known": True,
                "downstream_usage_total_known": True,
                "downstream_usage_values_valid": True,
                "downstream_usage_alias_consistent": True,
                "downstream_usage_seen": True,
            },
        },
    )

    summary = telemetry["logs"][0]["summary"]
    assert summary["prompt_tokens"] == "0"
    assert summary["completion_tokens"] == "0"
    assert summary["total_tokens"] == "0"


def test_responses_request_summary_is_never_sampled_away():
    client = FugueObservabilityClient(
        FugueObservabilityConfig(
            endpoint="https://observability.invalid",
            sample_rate=0.0,
        )
    )
    client._queue = asyncio.Queue(maxsize=10)

    client.emit_request(
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "wire_status_code": 200,
            "stream_outcome": "completed",
            "responses_stream_diagnostics": {
                "diagnosis": "responses_completed_with_usage",
                "semantic_status": "completed",
                "usage_seen": True,
            },
        }
    )
    assert client._queue.qsize() == 1

    client._queue.get_nowait()
    client._queue.task_done()
    client.emit_request(
        current_info={
            "endpoint": "GET /healthz",
            "status_code": 200,
            "stream_outcome": "completed",
        }
    )
    assert client._queue.qsize() == 0


def test_responses_delta_fast_path_counters_reach_request_summary():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "endpoint": "POST /v1/responses",
            "status_code": 200,
            "stream": True,
            "responses_delta_fast_path_candidates": 491,
            "responses_delta_fast_path_events": 487,
            "responses_delta_fast_path_fallbacks": 4,
            "responses_delta_fast_path_bytes": 249_164,
        },
        runtime_metrics={},
    )

    summary = telemetry["logs"][0]["summary"]
    assert summary["responses_delta_fast_path_candidates"] == "491"
    assert summary["responses_delta_fast_path_events"] == "487"
    assert summary["responses_delta_fast_path_fallbacks"] == "4"
    assert summary["responses_delta_fast_path_bytes"] == "249164"


def test_image_stream_terminal_contract_is_exported_without_payload_data():
    telemetry = build_uni_api_ember_request_telemetry(
        service_name="uni-api-ember",
        service_version="test",
        identity_attrs={"app_id": "app_123"},
        current_info={
            "request_id": "image-edit-observability",
            "endpoint": "POST /v1/images/edits",
            "status_code": 200,
            "wire_status_code": 200,
            "stream": True,
            "stream_outcome": "upstream_stream_abort",
            "postcommit_sse_protocol_error_isolated": True,
            "image_stream_diagnostics": {
                "contract_version": 1,
                "last_event_type": "image_edit.partial_image",
                "last_data_type": "image_edit.partial_image",
                "eof": True,
                "terminal_seen": False,
                "synthetic_terminal": True,
                "synthetic_terminal_type": "error",
                "payload": "must-not-be-exported",
            },
        },
    )

    request_log = telemetry["logs"][0]
    attributes = request_log["attributes"]
    summary = request_log["summary"]
    for values in (attributes, summary):
        assert values["postcommit_sse_protocol_error_isolated"] == "true"
        assert values["image_stream_contract_version"] == "1"
        assert values["image_stream_last_event_type"] == (
            "image_edit.partial_image"
        )
        assert values["image_stream_last_data_type"] == (
            "image_edit.partial_image"
        )
        assert values["image_stream_eof"] == "true"
        assert values["image_stream_terminal_seen"] == "false"
        assert values["image_stream_synthetic_terminal"] == "true"
        assert values["image_stream_synthetic_terminal_type"] == "error"
    assert "must-not-be-exported" not in json.dumps(telemetry)


def test_traceparent_is_inherited_and_forwarded():
    incoming = main._incoming_trace_context(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "tracestate": "vendor=value",
            "x-request-id": "legacy-request",
        }
    )

    trace = main.RequestTrace(
        trace_id=incoming["trace_id"],
        parent_span_id=incoming["parent_span_id"],
        trace_flags=incoming["trace_flags"],
        tracestate=incoming["tracestate"],
    )
    headers = main._trace_headers_for_upstream(
        {
            "trace_id": trace.trace_id,
            "request_id": "request_123",
            "trace": trace,
            "tracestate": trace.tracestate,
        }
    )

    assert incoming["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert incoming["parent_span_id"] == "00f067aa0ba902b7"
    assert incoming["x_request_id"] == "legacy-request"
    assert headers["x-request-id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert headers["tracestate"] == "vendor=value"
    assert headers["traceparent"].startswith("00-4bf92f3577b34da6a3ce929d0e0e4736-")


def test_missing_trace_headers_generate_w3c_trace_id():
    incoming = main._incoming_trace_context({})

    assert re.match(r"^[0-9a-f]{32}$", incoming["trace_id"])


def test_request_trace_uses_nonzero_ms_for_observed_stages(monkeypatch):
    trace = main.RequestTrace(trace_id="4bf92f3577b34da6a3ce929d0e0e4736")
    monkeypatch.setattr(main, "time", lambda: trace.started_at)

    trace.mark("request_received")
    trace.mark("upstream_first_chunk")

    spans = trace.snapshot()
    assert spans["request_received"] == 0
    assert spans["upstream_first_chunk"] == 1
