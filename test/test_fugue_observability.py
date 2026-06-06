import json
import re

import main
from fugue_observability import (
    build_uni_api_ember_request_telemetry,
    fugue_observability_config_from_env,
)


def test_fugue_observability_disabled_without_endpoint(monkeypatch):
    monkeypatch.delenv("FUGUE_OBSERVABILITY_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    config = fugue_observability_config_from_env(service_version="test")

    assert config.enabled is False


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
            "process_time": 1.25,
            "api_key": "sk-secret-api-key",
            "text": "this is request body content",
            "message_roles": "system/user",
            "role_counts": "system:1,user:1",
            "retry_count": 1,
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
                "upstream_send_start": 30,
                "upstream_headers_received": 90,
                "upstream_first_chunk": 140,
                "downstream_response_start": 145,
                "stream_end": 1250,
            },
        },
        runtime_metrics={
            "inflight_requests": 12,
            "waiting_first_byte": 4,
            "event_loop_lag_ms": 2,
            "upstream_pool_in_use": 3,
        },
    )

    serialized = json.dumps(telemetry, sort_keys=True)
    assert "sk-secret-api-key" not in serialized
    assert "this is request body content" not in serialized
    assert "api_key_hash" in serialized
    assert "system/user" in serialized

    stages = {
        event["attributes"]["stage"]
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

    for metric in telemetry["metrics"]:
        attrs = metric["attributes"]
        assert "trace_id" not in attrs
        assert "request_id" not in attrs
        assert "api_key_hash" not in attrs


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
