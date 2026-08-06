import asyncio
from types import SimpleNamespace

import h11
import httpx

from uni_api.observability.exceptions import exception_diagnostics
from uni_api.observability.upstream_transport import (
    UpstreamTransportDiagnostics,
    bind_upstream_transport_diagnostics,
    compose_httpcore_trace,
    reset_upstream_transport_diagnostics,
)
from uni_api.upstream.client_pool import ClientPool


def _wrapped_local_protocol_error(message: str) -> httpx.LocalProtocolError:
    try:
        raise h11.LocalProtocolError(message)
    except h11.LocalProtocolError as cause:
        try:
            raise httpx.LocalProtocolError(str(cause)) from cause
        except httpx.LocalProtocolError as exc:
            return exc


def test_exception_diagnostics_redacts_and_classifies_content_length():
    exc = _wrapped_local_protocol_error(
        "Too little data for declared Content-Length; "
        "Authorization: Bearer provider-secret; "
        "https://user:pass@example.test/path?token=provider-secret"
    )

    diagnostics = exception_diagnostics(exc)

    assert diagnostics["protocol_error_reason"] == "CONTENT_LENGTH_MISMATCH"
    assert diagnostics["exception_module"] == "httpx"
    assert "provider-secret" not in diagnostics["exception_chain_json"]
    assert "Authorization: [redacted]" in diagnostics["exception_chain_json"]
    assert "[redacted]@" in diagnostics["exception_chain_json"]


def test_httpcore_trace_preserves_raw_exception_and_failure_stage():
    async def run():
        entry = {}
        diagnostics = UpstreamTransportDiagnostics(entry)
        exc = _wrapped_local_protocol_error(
            "can't handle event type Request when role=CLIENT and state=DONE"
        )

        await diagnostics.httpcore_trace(
            "http11.send_request_headers.failed",
            {"exception": exc},
        )
        diagnostics.observe_exception(exc, client=SimpleNamespace())
        diagnostics.finalize("failed")

        assert entry["failure_stage"] == "send_headers"
        assert entry["protocol_error_reason"] == "SEND_HEADERS_ON_CLOSED"
        assert entry["httpcore_exception_type"] == "LocalProtocolError"
        assert entry["httpcore_exception_module"] == "httpx"
        assert "h11" in entry["httpcore_exception_chain_json"]
        assert "send_request_headers.failed" in entry["httpcore_events_json"]

    asyncio.run(run())


def test_httpcore_trace_records_tls_and_http_phase_timings():
    async def run():
        entry = {}
        diagnostics = UpstreamTransportDiagnostics(entry)
        for name in (
            "connection.start_tls.started",
            "connection.start_tls.complete",
            "http11.send_request_headers.started",
            "http11.send_request_headers.complete",
            "http11.send_request_body.started",
            "http11.send_request_body.complete",
            "http11.receive_response_headers.started",
            "http11.receive_response_headers.complete",
        ):
            await diagnostics.httpcore_trace(name, {})

        for phase in (
            "tls",
            "request_headers",
            "request_body",
            "response_headers",
        ):
            assert entry[f"transport_{phase}_started_ms"] >= 0
            assert entry[f"transport_{phase}_completed_ms"] >= 0
            assert entry[f"transport_{phase}_duration_ms"] >= 0
            assert entry[f"transport_{phase}_status"] == "complete"

    asyncio.run(run())


def test_response_body_observer_records_only_the_first_nonempty_chunk():
    class BodyStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b""
            yield b"abc"
            yield b"defgh"

        async def aclose(self):
            return None

    async def run():
        entry = {}
        diagnostics = UpstreamTransportDiagnostics(entry)
        response = httpx.Response(
            200,
            request=httpx.Request("GET", "http://test.local"),
            stream=BodyStream(),
        )
        diagnostics.capture_response(response, client=SimpleNamespace())
        assert b"".join([chunk async for chunk in response.aiter_bytes()]) == b"abcdefgh"
        diagnostics.capture_response(response, client=SimpleNamespace())

        assert entry["transport_first_body_ms"] >= 0
        assert entry["transport_first_body_bytes"] == 3

    asyncio.run(run())


def test_transport_diagnostics_classifies_pool_timeout_as_local_overload():
    entry = {}
    diagnostics = UpstreamTransportDiagnostics(entry)

    diagnostics.observe_exception(
        httpx.PoolTimeout("pool exhausted"),
        client=SimpleNamespace(),
    )
    diagnostics.finalize("failed")

    assert entry["transport_error_kind"] == "pool_timeout"
    assert entry["transport_error_owner"] == "ember_local_overload"
    assert entry["transport_error_phase"] == "pool_acquire"
    assert entry["transport_error_status_code"] == 503
    assert entry["provider_penalty_eligible"] is False
    assert entry["local_overload"] is True


def test_transport_diagnostics_classifies_read_timeout_before_headers():
    async def run():
        entry = {}
        diagnostics = UpstreamTransportDiagnostics(entry)
        await diagnostics.httpcore_trace(
            "http11.receive_response_headers.started",
            {},
        )

        diagnostics.observe_exception(
            httpx.ReadTimeout("read timed out"),
            client=SimpleNamespace(),
        )

        assert entry["transport_error_kind"] == "read_timeout"
        assert entry["transport_error_phase"] == "before_response_headers"

    asyncio.run(run())


def test_connection_snapshot_records_h2_state_goaway_and_alpn():
    class SSLObject:
        def selected_alpn_protocol(self):
            return "h2"

    class NetworkStream:
        def get_extra_info(self, name):
            return SSLObject() if name == "ssl_object" else None

    async def run():
        origin = object()
        network_stream = NetworkStream()
        stream_state = SimpleNamespace(
            state_machine=SimpleNamespace(state=SimpleNamespace(name="OPEN"))
        )
        h2_state = SimpleNamespace(
            state_machine=SimpleNamespace(
                state=SimpleNamespace(name="CLIENT_OPEN")
            ),
            streams={3: stream_state},
        )
        connection = SimpleNamespace(
            _origin=origin,
            _network_stream=network_stream,
            _events={3: []},
            _h2_state=h2_state,
            _state=SimpleNamespace(name="ACTIVE"),
            _request_count=17,
            _max_streams=100,
            _connection_error=True,
            _connection_terminated=SimpleNamespace(
                error_code=11,
                last_stream_id=3,
            ),
            _read_exception=RuntimeError("read failed"),
            _write_exception=None,
        )
        client = SimpleNamespace(
            _transport=SimpleNamespace(
                _pool=SimpleNamespace(
                    _connections=[SimpleNamespace(_connection=connection)]
                )
            ),
            _mounts={},
        )
        entry = {}
        diagnostics = UpstreamTransportDiagnostics(entry)
        await diagnostics.httpcore_trace(
            "http2.send_request_headers.started",
            {
                "request": SimpleNamespace(
                    url=SimpleNamespace(origin=origin)
                ),
                "stream_id": 3,
            },
        )
        diagnostics.capture_response(
            SimpleNamespace(
                status_code=200,
                http_version="HTTP/2",
                extensions={
                    "http_version": b"HTTP/2",
                    "stream_id": 3,
                    "network_stream": network_stream,
                },
                request=None,
            ),
            client=client,
        )

        assert diagnostics.facts["http_version"] == "HTTP/2"
        assert diagnostics.facts["alpn_protocol"] == "h2"
        assert diagnostics.facts["http2_stream_id"] == 3
        assert diagnostics.facts["connection_request_count"] == 17
        assert diagnostics.facts["http2_concurrent_streams"] == 1
        assert diagnostics.facts["http2_max_concurrent_streams"] == 100
        assert diagnostics.facts["http2_local_connection_state"] == "CLIENT_OPEN"
        assert diagnostics.facts["http2_local_stream_state"] == "OPEN"
        assert diagnostics.facts["goaway_error_code"] == 11
        assert diagnostics.facts["goaway_last_stream_id"] == 3

    asyncio.run(run())


def test_composed_trace_is_fail_open_and_keeps_existing_callback():
    async def run():
        entry = {}
        diagnostics = UpstreamTransportDiagnostics(entry)
        observed = []

        async def existing(name, _info):
            observed.append(name)
            raise RuntimeError("telemetry callback failure")

        callback = compose_httpcore_trace(existing, diagnostics)
        assert callback is not None
        await callback("http11.send_request_headers.started", {})

        assert observed == ["http11.send_request_headers.started"]
        assert diagnostics.facts["failure_stage"] == "send_headers"

    asyncio.run(run())


def test_httpcore_control_flow_close_is_not_reported_as_transport_failure():
    async def run():
        entry = {}
        diagnostics = UpstreamTransportDiagnostics(entry)

        await diagnostics.httpcore_trace(
            "http11.receive_response_body.failed",
            {"exception": GeneratorExit()},
        )
        diagnostics.finalize("stream_completed")

        assert entry["outcome"] == "stream_completed"
        assert "httpcore_exception_type" not in entry
        assert "control_flow_exception_type" in entry["httpcore_events_json"]

    asyncio.run(run())


def test_managed_client_records_real_http11_connection_metadata():
    async def handle(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 2\r\n"
                b"Connection: keep-alive\r\n\r\n{}"
            )
            await writer.drain()
            try:
                await asyncio.wait_for(reader.read(), timeout=1)
            except TimeoutError:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def run():
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        pool = ClientPool(pool_size=1, waiter_limit=0)
        await pool.init(
            {
                "http2": True,
                "verify": False,
                "follow_redirects": False,
            }
        )
        entry = {}
        diagnostics = UpstreamTransportDiagnostics(entry)
        token = bind_upstream_transport_diagnostics(diagnostics)
        try:
            async with pool.get_client(f"http://127.0.0.1:{port}") as client:
                response = await client.get(f"http://127.0.0.1:{port}/health")
            assert response.status_code == 200
            assert response.json() == {}
            diagnostics.finalize("succeeded")
        finally:
            reset_upstream_transport_diagnostics(token)
            await pool.close()
            server.close()
            await server.wait_closed()

        assert entry["http_version"] == "HTTP/1.1"
        assert entry["connection_protocol"] == "AsyncHTTP11Connection"
        assert entry["connection_request_count"] == 1
        assert entry["connection_snapshot_match"] == "exact"
        assert "send_request_headers.started" in entry["httpcore_events_json"]
        assert entry["transport_dns_status"] == "ip_literal"
        assert entry["transport_dns_duration_ms"] >= 0
        assert entry["transport_tcp_duration_ms"] >= 0
        assert entry["transport_request_headers_duration_ms"] >= 0
        assert entry["transport_request_body_duration_ms"] >= 0
        assert entry["transport_response_headers_duration_ms"] >= 0
        assert entry["transport_first_body_bytes"] == 2
        assert entry["transport_first_body_ms"] >= 0

    asyncio.run(run())
