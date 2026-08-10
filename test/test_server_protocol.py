from __future__ import annotations

import asyncio
import socket

import uvicorn

from uni_api.admission import RequestAdmissionController
from uni_api.disconnect import DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
from uni_api.middleware.admission import (
    RESERVE_BODY_BYTES_STATE_KEY,
    RequestAdmissionMiddleware,
)
from uni_api.server import build_bounded_h11_protocol


async def _instant_app(scope, receive, send):
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-length", b"2")],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})


async def _start_server(
    *,
    connection_limit: int | None,
    header_timeout: float,
    app=_instant_app,
    connection_admission=None,
):
    protocol, stats = build_bounded_h11_protocol(
        connection_limit=connection_limit,
        header_timeout_seconds=header_timeout,
        connection_admission=connection_admission,
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    config = uvicorn.Config(
        app,
        http=protocol,
        lifespan="off",
        limit_concurrency=None,
        log_level="error",
        timeout_keep_alive=10,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    while not server.started:
        await asyncio.sleep(0.001)
    return server, task, listener, port, stats


async def _stop_server(server, task, listener):
    server.should_exit = True
    await asyncio.wait_for(task, timeout=2)
    listener.close()


async def _wait_until(predicate, *, timeout: float = 1.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


def test_preconnected_keepalive_sockets_do_not_trigger_uvicorn_false_503s():
    async def scenario():
        server, task, listener, port, stats = await _start_server(
            connection_limit=10,
            header_timeout=1,
        )
        connections = []
        overflow_writer = None
        try:
            for _ in range(10):
                connections.append(
                    await asyncio.open_connection("127.0.0.1", port)
                )
            while stats.accepted_connections < 10:
                await asyncio.sleep(0.001)

            overflow_reader, overflow_writer = await asyncio.open_connection(
                "127.0.0.1",
                port,
            )
            assert await asyncio.wait_for(overflow_reader.read(), timeout=1) == b""
            assert stats.rejected_connections == 1

            for _reader, writer in connections:
                writer.write(
                    b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
                )
            await asyncio.gather(
                *(writer.drain() for _reader, writer in connections)
            )
            responses = await asyncio.gather(
                *(reader.read() for reader, _writer in connections)
            )
            assert all(b" 200 OK\r\n" in response for response in responses)
            assert all(b" 503 Service Unavailable\r\n" not in response for response in responses)
        finally:
            if overflow_writer is not None:
                overflow_writer.close()
            for _reader, writer in connections:
                writer.close()
            await _stop_server(server, task, listener)

    asyncio.run(scenario())


def test_resource_guard_replaces_static_connection_count_limit():
    async def scenario():
        decisions = iter((False, True))
        server, task, listener, port, stats = await _start_server(
            connection_limit=None,
            header_timeout=1,
            connection_admission=lambda: next(decisions),
        )
        first_writer = None
        second_writer = None
        try:
            first_reader, first_writer = await asyncio.open_connection(
                "127.0.0.1",
                port,
            )
            assert await asyncio.wait_for(first_reader.read(), timeout=1) == b""

            second_reader, second_writer = await asyncio.open_connection(
                "127.0.0.1",
                port,
            )
            second_writer.write(
                b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
            )
            await second_writer.drain()
            response = await asyncio.wait_for(second_reader.read(), timeout=1)
            assert b" 200 OK\r\n" in response
            assert stats.connection_limit is None
            assert stats.rejected_connections == 0
            assert stats.resource_rejected_connections == 1
        finally:
            if first_writer is not None:
                first_writer.close()
            if second_writer is not None:
                second_writer.close()
            await _stop_server(server, task, listener)

    asyncio.run(scenario())


def test_incomplete_request_header_is_closed_at_absolute_deadline():
    async def scenario():
        server, task, listener, port, stats = await _start_server(
            connection_limit=2,
            header_timeout=0.05,
        )
        writer = None
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET / HTTP/1.1\r\nHost:")
            await writer.drain()
            assert await asyncio.wait_for(reader.read(), timeout=1) == b""
            assert stats.header_timeouts == 1
        finally:
            if writer is not None:
                writer.close()
            await _stop_server(server, task, listener)

    asyncio.run(scenario())


def test_fragmented_body_is_backpressured_then_completes_after_admission():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=1,
            wait_timeout_seconds=2,
            max_body_bytes=64,
            body_budget_bytes=64,
        )
        holder = await controller.acquire()
        received_bodies = []

        async def body_app(scope, receive, send):
            body = bytearray()
            reserve_body_bytes = scope["state"][RESERVE_BODY_BYTES_STATE_KEY]
            while True:
                message = await receive()
                chunk = message.get("body", b"") or b""
                await reserve_body_bytes(len(chunk))
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            received_bodies.append(bytes(body))
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-length", b"2")],
                }
            )
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = RequestAdmissionMiddleware(body_app, controller=controller)
        server, task, listener, port, _stats = await _start_server(
            connection_limit=4,
            header_timeout=1,
            app=middleware,
        )
        writer = None
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"POST /v1/responses HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Length: 32\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
            for expected_pending in range(1, 17):
                writer.write(b"x")
                await writer.drain()
                await _wait_until(
                    lambda expected=expected_pending: controller.snapshot()[
                        "pending_body_reserved_bytes"
                    ]
                    == expected
                )

            writer.write(b"x" * 16)
            await writer.drain()
            await asyncio.sleep(0.05)
            snapshot = controller.snapshot()
            assert snapshot["waiters"] == 1
            assert snapshot["pending_body_reserved_bytes"] == 16

            await holder.release()
            response = await asyncio.wait_for(reader.read(), timeout=2)
            assert b" 200 OK\r\n" in response
            assert received_bodies == [b"x" * 32]
            snapshot = controller.snapshot()
            assert snapshot["active"] == 0
            assert snapshot["waiters"] == 0
            assert snapshot["reserved_body_bytes"] == 0
            assert snapshot["pending_body_reserved_bytes"] == 0
            assert snapshot["rejected"] == {}
        finally:
            if not holder.released:
                await holder.release()
            if writer is not None:
                writer.close()
            await _stop_server(server, task, listener)

    asyncio.run(scenario())


def test_transport_close_releases_a_backpressured_waiter_immediately():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=1,
            wait_timeout_seconds=2,
            max_body_bytes=64,
            body_budget_bytes=64,
        )
        holder = await controller.acquire()
        app_called = False
        observed = []

        async def body_app(scope, receive, send):
            nonlocal app_called
            app_called = True

        async def on_early(scope, status_code, reason):
            observed.append((status_code, reason))

        middleware = RequestAdmissionMiddleware(
            body_app,
            controller=controller,
            on_early_response=on_early,
        )
        server, task, listener, port, _stats = await _start_server(
            connection_limit=4,
            header_timeout=1,
            app=middleware,
        )
        writer = None
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"POST /v1/responses HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Length: 32\r\n\r\n"
            )
            await writer.drain()
            for expected_pending in range(1, 17):
                writer.write(b"x")
                await writer.drain()
                await _wait_until(
                    lambda expected=expected_pending: controller.snapshot()[
                        "pending_body_reserved_bytes"
                    ]
                    == expected
                )

            writer.close()
            await writer.wait_closed()
            writer = None
            await _wait_until(
                lambda: observed
                == [(499, "disconnected_while_queued")],
                timeout=0.5,
            )

            snapshot = controller.snapshot()
            assert app_called is False
            assert observed == [(499, "disconnected_while_queued")]
            assert snapshot["active"] == 1
            assert snapshot["pending_body_reserved_bytes"] == 0
            assert snapshot["reserved_body_bytes"] == 0
            assert snapshot["rejected"] == {}
        finally:
            if writer is not None:
                writer.close()
            await holder.release()
            await _stop_server(server, task, listener)

    asyncio.run(scenario())


def test_pipelined_requests_get_distinct_events_and_normal_close_sets_neither():
    async def scenario():
        observed_events = []

        async def event_app(scope, receive, send):
            observed_events.append(
                scope["state"][DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY]
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-length", b"2")],
                }
            )
            await send({"type": "http.response.body", "body": b"ok"})

        server, task, listener, port, _stats = await _start_server(
            connection_limit=2,
            header_timeout=1,
            app=event_app,
        )
        writer = None
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"GET /first HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"\r\n"
                b"GET /second HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), timeout=1)
            assert response.count(b" 200 OK\r\n") == 2
            await asyncio.sleep(0)
            assert len(observed_events) == 2
            assert observed_events[0] is not observed_events[1]
            assert all(event.is_set() is False for event in observed_events)
        finally:
            if writer is not None:
                writer.close()
            await _stop_server(server, task, listener)

    asyncio.run(scenario())
