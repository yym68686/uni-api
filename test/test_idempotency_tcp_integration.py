import asyncio
import socket

import httpx
import uvicorn

from uni_api.admission import RequestAdmissionController
from uni_api.disconnect import DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
from uni_api.middleware.admission import RequestAdmissionMiddleware
from uni_api.middleware.idempotency import (
    IdempotencyMiddleware,
    InMemoryIdempotencyCoordinator,
)
from uni_api.middleware.request_decompression import (
    RequestBodyDecompressionMiddleware,
)
from uni_api.server import build_bounded_h11_protocol


def test_real_tcp_disconnect_finishes_owner_and_replays_on_new_connection():
    async def scenario():
        calls = 0
        first_chunk_sent = asyncio.Event()
        finish_stream = asyncio.Event()
        transport_events: list[asyncio.Event] = []

        async def streaming_app(scope, receive, send):
            nonlocal calls
            calls += 1
            request = await receive()
            assert request["body"] == b'{"model":"gpt-test"}'
            detached_event = scope["state"][
                DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
            ]
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"data: first\n\n",
                    "more_body": True,
                }
            )
            first_chunk_sent.set()
            await finish_stream.wait()
            assert detached_event.is_set() is False
            await send(
                {
                    "type": "http.response.body",
                    "body": b"data: done\n\n",
                    "more_body": False,
                }
            )

        coordinator = InMemoryIdempotencyCoordinator(
            ttl_seconds=60,
            max_entries=32,
            max_stored_bytes=1024 * 1024,
            max_response_bytes=1024 * 1024,
        )
        controller = RequestAdmissionController(
            capacity=4,
            waiter_limit=4,
            wait_timeout_seconds=1,
            max_body_bytes=1024 * 1024,
            body_budget_bytes=4 * 1024 * 1024,
            max_response_bytes=1024 * 1024,
        )
        inner = RequestAdmissionMiddleware(
            IdempotencyMiddleware(
                RequestBodyDecompressionMiddleware(
                    streaming_app,
                    max_identity_body_bytes=1024 * 1024,
                ),
                coordinator=coordinator,
            ),
            controller=controller,
        )

        async def capture_transport_event(scope, receive, send):
            event = scope["state"][DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY]
            transport_events.append(event)
            await inner(scope, receive, send)

        protocol, _protocol_stats = build_bounded_h11_protocol(
            connection_limit=8,
            header_timeout_seconds=2,
        )
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(64)
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                capture_transport_event,
                http=protocol,
                lifespan="off",
                limit_concurrency=None,
                log_level="critical",
                timeout_keep_alive=10,
            )
        )
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        writer = None
        try:
            while not server.started:
                await asyncio.sleep(0.001)

            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            body = b'{"model":"gpt-test"}'
            writer.write(
                b"POST /v1/responses HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Authorization: Bearer client-a\r\n"
                b"Content-Type: application/json\r\n"
                b"Idempotency-Key: logical-request-1\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                + body
            )
            await writer.drain()
            response_prefix = await asyncio.wait_for(
                reader.readuntil(b"data: first\n\n"),
                timeout=2,
            )
            assert b"HTTP/1.1 200 OK" in response_prefix
            await asyncio.wait_for(first_chunk_sent.wait(), timeout=1)

            writer.close()
            await writer.wait_closed()
            writer = None
            assert transport_events
            await asyncio.wait_for(transport_events[0].wait(), timeout=1)
            finish_stream.set()

            async with asyncio.timeout(2):
                while coordinator.snapshot()["completed"] != 1:
                    await asyncio.sleep(0.001)

            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                trust_env=False,
                timeout=2,
            ) as client:
                replay = await client.post(
                    "/v1/responses",
                    headers={
                        "Authorization": "Bearer client-a",
                        "Content-Type": "application/json",
                        "Idempotency-Key": "logical-request-1",
                    },
                    content=body,
                )

            assert replay.status_code == 200
            assert replay.headers["x-uni-api-idempotency-status"] == "replayed"
            assert replay.content == b"data: first\n\ndata: done\n\n"
            assert calls == 1
            snapshot = coordinator.snapshot()
            assert snapshot["downstream_disconnects_detached"] == 1
            async with asyncio.timeout(1):
                while controller.snapshot()["active"] != 0:
                    await asyncio.sleep(0)
            assert controller.snapshot()["active"] == 0
        finally:
            if writer is not None:
                writer.close()
            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=5)
            listener.close()

    asyncio.run(scenario())
