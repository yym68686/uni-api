from datetime import datetime, timezone
from time import time
from types import SimpleNamespace

from starlette.responses import StreamingResponse

from uni_api.observability.request_context import (
    RequestContext,
    get_request_info,
    request_info,
    reset_request_info,
    set_request_info,
)
from uni_api.api.health import observability_runtime_response
from uni_api.observability.paid_keys import compute_paid_api_key_state
from uni_api.observability.telemetry import emit_request_observability


def test_request_context_wraps_shared_contextvar():
    token = set_request_info({"request_id": "req-1"})
    try:
        assert get_request_info()["request_id"] == "req-1"
        assert request_info.get()["request_id"] == "req-1"
    finally:
        reset_request_info(token)


def test_request_context_round_trips_known_fields_and_extras():
    context = RequestContext(
        request_id="req-1",
        trace_id="trace-1",
        endpoint="POST /v1/chat/completions",
        api_key="sk-test",
        extras={"trace": object(), "stream": True},
    )

    payload = context.to_dict()
    restored = RequestContext.from_dict(payload)

    assert restored.request_id == "req-1"
    assert restored.endpoint == "POST /v1/chat/completions"
    assert restored.extras["stream"] is True
    assert "trace" in restored.extras


def test_telemetry_adapter_delegates_to_fugue_emit(monkeypatch):
    calls = []

    def fake_emit(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("uni_api.observability.telemetry.emit_uni_api_ember_request_observability", fake_emit)

    emit_request_observability({"request_id": "req-1"}, {"inflight_requests": 1})

    assert calls == [
        {
            "current_info": {"request_id": "req-1"},
            "runtime_metrics": {"inflight_requests": 1},
        }
    ]


def test_debug_header_pairs_masks_sensitive_headers():
    import main

    pairs = main._debug_header_pairs(
        {
            "Authorization": "Bearer sk-test-secret-value",
            "x-api-key": "sk-test-secret-value",
            "X-Trace-Id": "trace-1",
        }
    )

    values = {item["name"].lower(): item["value"] for item in pairs}
    assert values["authorization"] == "Bear...alue"
    assert values["x-api-key"] == "sk-t...alue"
    assert values["x-trace-id"] == "trace-1"
    assert "sk-test-secret-value" not in str(pairs)


async def test_observability_runtime_includes_background_task_snapshots():
    class RuntimeGauges:
        async def record_event_loop_lag(self):
            return None

        def snapshot(self):
            return {"event_loop_lag_ms": 0}

    payload = await observability_runtime_response(
        RuntimeGauges(),
        stream_cleanup_snapshot=lambda: {"pending": 1, "done": 0, "total": 1},
        provider_key_pools_snapshot=lambda: {"total": 2, "reordering_task_active": 1},
    )

    assert payload["stream_cleanup_tasks"]["pending"] == 1
    assert payload["provider_key_pools"]["reordering_task_active"] == 1


async def test_paid_api_key_state_computes_enabled_from_cost_and_credits():
    async def fake_total_cost(**kwargs):
        assert kwargs["filter_api_key"] == "sk-paid"
        return 0.75

    async def fake_usage(**kwargs):
        assert kwargs["filter_api_key"] == "sk-paid"
        return [{"model": "gpt-4.1", "total_tokens": 10}]

    state, total_cost = await compute_paid_api_key_state(
        credits=1.0,
        created_at=datetime.now(timezone.utc),
        paid_key="sk-paid",
        compute_total_cost=fake_total_cost,
        get_usage_data=fake_usage,
    )

    assert total_cost == 0.75
    assert state is not None
    assert state.enabled is True
    assert state.to_dict()["all_tokens_info"][0]["total_tokens"] == 10


async def test_streaming_observability_wrap_merges_response_current_info():
    import main

    async def update_stats(info):
        _ = info

    outer_trace = main.RequestTrace(trace_id="outer")
    inner_trace = main.RequestTrace(trace_id="inner")
    inner_trace.mark("provider_selected")
    inner_info = {
        "request_id": "inner-req",
        "api_key": "sk-test",
        "trace": inner_trace,
        "timing_spans": inner_trace.snapshot(),
    }
    outer_info = {
        "request_id": "outer-req",
        "api_key": "sk-test",
        "trace": outer_trace,
        "timing_spans": outer_trace.snapshot(),
    }
    response = StreamingResponse(iter([b"data: ok\n\n"]), media_type="text/event-stream")
    response.current_info = inner_info
    middleware = object.__new__(main.StatsMiddleware)
    middleware.dependencies = SimpleNamespace(
        database_disabled=False,
        logging_response_class=main.LoggingStreamingResponse,
        mark_first_byte_observed=lambda info: None,
        emit_request_observability=lambda info: None,
        update_stats=update_stats,
        debug=False,
    )

    wrapped = await middleware._wrap_response_for_observability(
        SimpleNamespace(url=SimpleNamespace(path="/v1/responses")),
        response,
        outer_info,
        outer_trace,
    )

    assert wrapped.current_info is outer_info
    assert outer_info["request_id"] == "inner-req"
    assert outer_info["trace"] is inner_trace
    assert outer_info["timing_spans"]["provider_selected"] >= 1
    assert wrapped.headers["x-request-id"] == "inner"


async def test_logging_streaming_response_preserves_handler_timing_spans():
    import main

    async def body():
        yield b"data: ok\n\n"

    emitted = []
    updated = []
    trace = main.RequestTrace(trace_id="trace-stream")
    trace.mark("request_received")
    current_info = {
        "request_id": "req-stream",
        "trace_id": "trace-stream",
        "endpoint": "POST /v1/responses",
        "api_key": "sk-test",
        "start_time": time(),
        "trace": trace,
        "timing_spans": {
            "request_received": 0,
            "provider_selected": 12,
            "provider_key_selected": 14,
            "upstream_send_start": 20,
            "upstream_headers_received": 45,
            "upstream_first_chunk": 80,
        },
    }

    async def update_stats(info):
        updated.append(dict(info))

    response = main.LoggingStreamingResponse(
        body(),
        status_code=200,
        media_type="text/event-stream",
        current_info=current_info,
        mark_first_byte_observed=lambda info: None,
        emit_request_observability=lambda info: emitted.append(dict(info)),
        update_stats=update_stats,
        trace_type=main.RequestTrace,
    )
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await response({"type": "http", "method": "POST", "path": "/v1/responses"}, receive, send)

    spans = emitted[0]["timing_spans"]
    assert spans["provider_selected"] == 12
    assert spans["provider_key_selected"] == 14
    assert spans["upstream_send_start"] == 20
    assert spans["upstream_headers_received"] == 45
    assert spans["upstream_first_chunk"] == 80
    assert spans["downstream_response_start"] >= 1
    assert spans["stream_end"] >= 1
    assert updated
    assert sent[-1] == {"type": "http.response.body", "body": b"", "more_body": False}
