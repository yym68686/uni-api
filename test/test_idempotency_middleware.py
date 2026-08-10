import asyncio
import hashlib
import json
import os
import stat
import threading

from uni_api.admission import RequestAdmissionController
from uni_api.admission.memory import AdaptiveMemoryGovernor, ProcessMemorySample
from uni_api.disconnect import DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
from uni_api.middleware.admission import RequestAdmissionMiddleware
from uni_api.middleware.idempotency import (
    IdempotencyMiddleware,
    InMemoryIdempotencyCoordinator,
    _request_identities,
)
from uni_api.middleware.request_decompression import (
    RequestBodyDecompressionMiddleware,
)


def _scope(
    *,
    key: str | None = "logical-request-1",
    authorization: str = "Bearer client-a",
    disconnect_event: asyncio.Event | None = None,
) -> dict:
    headers = [
        (b"authorization", authorization.encode("ascii")),
        (b"content-type", b"application/json"),
    ]
    if key is not None:
        headers.append((b"idempotency-key", key.encode("ascii")))
    state = {}
    if disconnect_event is not None:
        state[DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY] = disconnect_event
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": state,
    }


def _receive_body(body: bytes):
    sent = False
    blocked = asyncio.Event()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        await blocked.wait()
        return {"type": "http.disconnect"}

    return receive


async def _invoke(
    middleware,
    *,
    body: bytes = b'{"model":"gpt-test"}',
    scope: dict | None = None,
) -> list[dict]:
    sent: list[dict] = []

    async def send(message):
        sent.append(dict(message))

    await middleware(scope or _scope(), _receive_body(body), send)
    return sent


def _response(sent: list[dict]) -> tuple[int, bytes, dict[bytes, bytes]]:
    start = next(item for item in sent if item["type"] == "http.response.start")
    body = b"".join(
        item.get("body", b"")
        for item in sent
        if item["type"] == "http.response.body"
    )
    return start["status"], body, dict(start["headers"])


def _coordinator(**overrides) -> InMemoryIdempotencyCoordinator:
    settings = {
        "ttl_seconds": 60,
        "max_entries": 32,
        "max_stored_bytes": 1024 * 1024,
        "max_response_bytes": 1024 * 1024,
    }
    settings.update(overrides)
    return InMemoryIdempotencyCoordinator(**settings)


def test_incremental_request_identity_matches_legacy_wire_identity():
    scope = _scope(authorization="Bearer client-a")
    scope["method"] = "p\N{LATIN SMALL LETTER O WITH DIAERESIS}st"
    scope["path"] = "/v1/r\N{LATIN SMALL LETTER E WITH ACUTE}sponses"
    scope["query_string"] = b"cursor=a%00b"
    scope["headers"].extend(
        [
            (b"x-api-key", b"secondary-key"),
            (b"content-type", b"charset=utf-8"),
            (b"content-encoding", b"zstd"),
        ]
    )
    key = "logical-request-1"
    body = (b'{}\x00{"input":"payload"}' * 4096)

    record_key, request_hash, key_fingerprint = _request_identities(
        scope,
        key,
        body,
    )

    headers: dict[str, list[str]] = {}
    for name, value in scope["headers"]:
        headers.setdefault(name.decode("latin-1").lower(), []).append(
            value.decode("latin-1")
        )
    joined_headers = {
        name: "\n".join(values) for name, values in headers.items()
    }
    method = str(scope["method"]).upper()
    path = str(scope["path"])
    query = bytes(scope["query_string"])
    credential = "\n".join(
        joined_headers.get(name, "")
        for name in ("authorization", "x-api-key")
    )
    credential_hash = hashlib.sha256(credential.encode("utf-8")).hexdigest()
    expected_record_key = hashlib.sha256(
        b"\x00".join(
            (
                method.encode("ascii", errors="replace"),
                path.encode("utf-8"),
                query,
                credential_hash.encode("ascii"),
                key.encode("ascii"),
            )
        )
    ).hexdigest()
    expected_request_hash = hashlib.sha256(
        b"\x00".join(
            (
                method.encode("ascii", errors="replace"),
                path.encode("utf-8"),
                query,
                joined_headers["content-type"].encode("latin-1"),
                joined_headers["content-encoding"].encode("latin-1"),
                body,
            )
        )
    ).hexdigest()

    assert record_key == expected_record_key
    assert request_hash == expected_request_hash
    assert key_fingerprint == hashlib.sha256(key.encode("ascii")).hexdigest()[
        :16
    ]


def test_idempotency_hash_phase_sampling_reports_only_numeric_cost():
    async def run():
        observed = []

        async def app(_scope, receive, send):
            request = await receive()
            assert request["body"] == b'{"model":"gpt-test"}'
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"ok",
                    "more_body": False,
                }
            )

        middleware = IdempotencyMiddleware(
            app,
            coordinator=_coordinator(),
            phase_sample_decider=lambda: True,
            phase_observer=lambda phase, **metrics: observed.append(
                (phase, metrics)
            ),
        )
        await _invoke(middleware)
        return observed

    observed = asyncio.run(run())
    assert len(observed) == 1
    phase, metrics = observed[0]
    assert phase == "idempotency_hash"
    assert metrics["bytes_count"] == len(b'{"model":"gpt-test"}')
    assert metrics["events"] == 1
    assert metrics["wall_ns"] >= 0
    assert metrics["cpu_ns"] >= 0
    assert all(isinstance(value, int) for value in metrics.values())


def test_explicit_key_executes_once_and_replays_completed_response():
    async def run():
        calls = 0

        async def app(scope, receive, send):
            nonlocal calls
            calls += 1
            assert all(
                name.lower() != b"idempotency-key"
                for name, _value in scope["headers"]
            )
            request = await receive()
            assert request["body"] == b'{"model":"gpt-test"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 201,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"id":"response-1"}',
                    "more_body": False,
                }
            )

        coordinator = _coordinator()
        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        first = await _invoke(middleware)
        second = await _invoke(middleware)

        assert calls == 1
        assert _response(first) == (
            201,
            b'{"id":"response-1"}',
            {
                b"content-type": b"application/json",
                b"x-uni-api-idempotency-status": b"executed",
            },
        )
        assert _response(second) == (
            201,
            b'{"id":"response-1"}',
            {
                b"content-type": b"application/json",
                b"x-uni-api-idempotency-status": b"replayed",
            },
        )
        snapshot = coordinator.snapshot()
        assert snapshot["owners"] == 1
        assert snapshot["replays"] == 1
        assert snapshot["mode"] == "spool-single-process"
        assert snapshot["persistence"] is False

    asyncio.run(run())


def test_concurrent_duplicate_waits_for_single_owner_then_replays():
    async def run():
        calls = 0
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def app(scope, receive, send):
            nonlocal calls
            _ = scope, receive
            calls += 1
            entered.set()
            await finish.wait()
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"done",
                    "more_body": False,
                }
            )

        coordinator = _coordinator()
        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        owner_task = asyncio.create_task(_invoke(middleware))
        await entered.wait()
        duplicate_task = asyncio.create_task(_invoke(middleware))
        async with asyncio.timeout(1.0):
            while coordinator.snapshot()["waits"] != 1:
                await asyncio.sleep(0)
        assert calls == 1
        finish.set()
        owner, duplicate = await asyncio.gather(owner_task, duplicate_task)

        assert calls == 1
        assert _response(owner)[2][b"x-uni-api-idempotency-status"] == b"executed"
        assert _response(duplicate)[2][b"x-uni-api-idempotency-status"] == b"replayed"
        assert coordinator.snapshot()["waits"] == 1

    asyncio.run(run())


def test_request_spool_is_released_as_soon_as_downstream_consumes_body():
    async def run():
        consumed = asyncio.Event()
        finish = asyncio.Event()

        coordinator = _coordinator(spool_memory_threshold_bytes=8)

        async def app(_scope, receive, send):
            chunks = []
            while True:
                message = await receive()
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            assert b"".join(chunks) == b"x" * 4096
            consumed.set()
            await finish.wait()
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {"type": "http.response.body", "body": b"ok", "more_body": False}
            )

        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        task = asyncio.create_task(_invoke(middleware, body=b"x" * 4096))
        await consumed.wait()
        snapshot = coordinator.snapshot()
        assert snapshot["spool_bytes_by_kind"].get("request_body", 0) == 0
        assert snapshot["spool_memory_bytes"] == 0
        finish.set()
        await task

    asyncio.run(run())


def test_disk_spool_write_does_not_block_the_event_loop():
    async def run():
        coordinator = _coordinator(max_stored_bytes=2 * 1024 * 1024)
        spool = coordinator.create_spool("inflight_response")
        original_write = spool._write_all
        entered = threading.Event()
        release = threading.Event()
        fallback = threading.Timer(0.5, release.set)

        def blocked_write(payload):
            entered.set()
            assert release.wait(timeout=1.0)
            original_write(payload)

        spool._write_all = blocked_write
        loop = asyncio.get_running_loop()
        loop.call_later(0.01, release.set)
        fallback.start()
        started = loop.time()
        try:
            assert await coordinator.append_spool(
                spool,
                b"x" * (256 * 1024),
            )
            assert entered.is_set()
            assert loop.time() - started < 0.2
        finally:
            fallback.cancel()
            fallback.join(timeout=1.0)
            await coordinator.close_spool(spool)

    asyncio.run(run())


def test_stream_chunks_are_coalesced_into_bounded_disk_writes():
    async def run():
        coordinator = _coordinator(max_stored_bytes=2 * 1024 * 1024)
        spool = coordinator.create_spool("inflight_response")
        original_write = spool._write_all
        write_sizes = []

        def observed_write(payload):
            write_sizes.append(len(payload))
            original_write(payload)

        spool._write_all = observed_write
        try:
            for _ in range(256):
                assert await coordinator.append_spool(spool, b"x" * 4096)
            await spool.seal_async(force_disk=True)
            assert write_sizes == [256 * 1024] * 4
            assert b"".join(spool.iter_chunks()) == b"x" * (1024 * 1024)
        finally:
            await coordinator.close_spool(spool)

    asyncio.run(run())


def test_append_skips_expiry_scan_without_spool_capacity_pressure():
    async def run():
        coordinator = _coordinator(max_stored_bytes=2 * 1024 * 1024)
        spool = coordinator.create_spool("inflight_response")
        original_prune = coordinator._prune_expired_locked
        prune_calls = 0

        def observed_prune(now):
            nonlocal prune_calls
            prune_calls += 1
            original_prune(now)

        coordinator._prune_expired_locked = observed_prune
        try:
            for _ in range(128):
                assert await coordinator.append_spool(spool, b"x" * 4096)
            assert prune_calls == 0
        finally:
            await coordinator.close_spool(spool)

    asyncio.run(run())


def test_in_memory_spool_prefix_is_charged_to_adaptive_governor():
    async def run():
        entered = asyncio.Event()
        consume = asyncio.Event()
        governor = AdaptiveMemoryGovernor(
            source=lambda: ProcessMemorySample(0, 1024 * 1024, source="test"),
            guard_bytes=0,
            guard_ratio=0,
            sample_cache_seconds=60,
        )
        coordinator = _coordinator(
            spool_memory_threshold_bytes=1024,
            memory_governor=governor,
        )

        async def app(_scope, receive, send):
            entered.set()
            await consume.wait()
            await receive()
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {"type": "http.response.body", "body": b"ok", "more_body": False}
            )

        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        task = asyncio.create_task(_invoke(middleware, body=b"x" * 128))
        await entered.wait()
        assert governor.snapshot().reservations == {
            "idempotency_request_body": 128
        }
        consume.set()
        await task
        assert governor.snapshot().reserved_bytes == 0

    asyncio.run(run())


def test_completed_response_cache_is_disk_backed_and_heap_bounded():
    async def run():
        async def app(_scope, _receive, send):
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"r" * 4096,
                    "more_body": False,
                }
            )

        coordinator = _coordinator(spool_memory_threshold_bytes=64)
        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        first = await _invoke(middleware)
        replay = await _invoke(middleware)

        assert _response(first)[1] == b"r" * 4096
        assert _response(replay)[1] == b"r" * 4096
        snapshot = coordinator.snapshot()
        assert snapshot["stored_response_bytes"] == 4096
        assert snapshot["spool_memory_bytes"] == 0
        assert snapshot["spool_bytes_by_kind"] == {
            "completed_response": 4096
        }
        entry = next(iter(coordinator._entries.values()))
        assert entry.response is not None
        spool_path = entry.response.body._path
        assert spool_path is not None
        assert stat.S_IMODE(os.stat(spool_path).st_mode) == 0o600
        assert stat.S_IMODE(
            os.stat(coordinator._spool_directory.name).st_mode
        ) == 0o700

    asyncio.run(run())


def test_spool_quota_failure_preserves_execution_tombstone():
    async def run():
        calls = 0

        async def app(_scope, receive, send):
            nonlocal calls
            calls += 1
            await receive()
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"response-too-large-for-global-spool",
                    "more_body": False,
                }
            )

        coordinator = _coordinator(
            max_stored_bytes=24,
            max_response_bytes=1024,
            spool_memory_threshold_bytes=8,
        )
        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        executed = await _invoke(middleware, body=b"{}")
        retry = await _invoke(middleware, body=b"{}")

        assert _response(executed)[0] == 200
        assert _response(retry)[0] == 409
        assert calls == 1
        snapshot = coordinator.snapshot()
        assert snapshot["spool_capacity_rejections"] == 1
        assert snapshot["nonreplayable_completed"] == 1
        assert snapshot["spool_total_bytes"] == 0

    asyncio.run(run())


def test_active_replay_pins_disk_entry_against_quota_eviction():
    async def run():
        async def app(_scope, receive, send):
            await receive()
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"12345678",
                    "more_body": False,
                }
            )

        coordinator = _coordinator(
            max_stored_bytes=8,
            max_response_bytes=8,
        )
        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        await _invoke(middleware, body=b"", scope=_scope(key="first"))

        replay_started = asyncio.Event()
        finish_replay = asyncio.Event()
        replay_messages = []

        async def blocked_send(message):
            replay_messages.append(dict(message))
            if message["type"] == "http.response.start":
                replay_started.set()
                await finish_replay.wait()

        replay_task = asyncio.create_task(
            middleware(
                _scope(key="first"),
                _receive_body(b""),
                blocked_send,
            )
        )
        await replay_started.wait()
        second = await _invoke(
            middleware,
            body=b"",
            scope=_scope(key="second"),
        )
        assert _response(second)[0] == 200
        assert coordinator.snapshot()["nonreplayable_completed"] == 1

        finish_replay.set()
        await replay_task
        assert _response(replay_messages)[1] == b"12345678"
        assert coordinator.snapshot()["replays"] == 1

    asyncio.run(run())


def test_same_key_with_different_body_is_rejected_without_second_execution():
    async def run():
        calls = 0

        async def app(scope, receive, send):
            nonlocal calls
            _ = scope, receive
            calls += 1
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {"type": "http.response.body", "body": b"ok", "more_body": False}
            )

        middleware = IdempotencyMiddleware(app, coordinator=_coordinator())
        await _invoke(middleware, body=b'{"input":"one"}')
        conflict = await _invoke(middleware, body=b'{"input":"two"}')

        status, body, headers = _response(conflict)
        assert calls == 1
        assert status == 409
        assert json.loads(body)["error"]["code"] == "conflict"
        assert headers[b"x-uni-api-idempotency-status"] == b"conflict"

    asyncio.run(run())


def test_key_scope_includes_authenticated_caller():
    async def run():
        calls = 0

        async def app(scope, receive, send):
            nonlocal calls
            _ = scope, receive
            calls += 1
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": str(calls).encode("ascii"),
                    "more_body": False,
                }
            )

        middleware = IdempotencyMiddleware(app, coordinator=_coordinator())
        first = await _invoke(
            middleware,
            scope=_scope(authorization="Bearer client-a"),
        )
        second = await _invoke(
            middleware,
            scope=_scope(authorization="Bearer client-b"),
        )

        assert calls == 2
        assert _response(first)[1] == b"1"
        assert _response(second)[1] == b"2"

    asyncio.run(run())


def test_transient_5xx_retries_but_oversized_success_gets_tombstone():
    async def run():
        calls = 0
        response_status = 503
        response_body = b"temporary"

        async def app(scope, receive, send):
            nonlocal calls
            _ = scope, receive
            calls += 1
            await send(
                {
                    "type": "http.response.start",
                    "status": response_status,
                    "headers": [],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": response_body,
                    "more_body": False,
                }
            )

        coordinator = _coordinator(max_response_bytes=4)
        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        await _invoke(middleware)
        await _invoke(middleware)
        assert calls == 2

        response_status = 200
        response_body = b"12345"
        executed = await _invoke(middleware, scope=_scope(key="oversized"))
        retry = await _invoke(middleware, scope=_scope(key="oversized"))
        assert calls == 3
        assert _response(executed)[0:2] == (200, b"12345")
        status, body, headers = _response(retry)
        assert status == 409
        assert json.loads(body)["error"]["code"] == "executed_nonreplayable"
        assert (
            headers[b"x-uni-api-idempotency-status"]
            == b"executed-nonreplayable"
        )
        snapshot = coordinator.snapshot()
        assert snapshot["responses_not_cached"] == 3
        assert snapshot["nonreplayable_completions"] == 1
        assert snapshot["nonreplayable_rejections"] == 1

    asyncio.run(run())


def test_incomplete_postcommit_stream_is_not_cached_for_idempotency_retry():
    async def run():
        calls = 0

        async def app(scope, receive, send):
            nonlocal calls
            _ = receive
            calls += 1
            current_info = scope["state"].setdefault(
                "uni_api_request_info",
                {},
            )
            current_info.update(
                {
                    "success": False,
                    "stream_outcome": "upstream_stream_abort",
                    "stream_error_after_response_start": True,
                    "postcommit_stream_terminal_suppressed": True,
                }
            )
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"event: response.output_text.delta\n\n",
                    "more_body": False,
                }
            )

        coordinator = _coordinator()
        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        first = await _invoke(middleware, scope=_scope(key="incomplete"))
        second = await _invoke(middleware, scope=_scope(key="incomplete"))

        assert calls == 2
        assert _response(first)[1] == b"event: response.output_text.delta\n\n"
        assert _response(second)[1] == b"event: response.output_text.delta\n\n"
        assert coordinator.snapshot()["responses_not_cached"] == 2

    asyncio.run(run())


def test_owner_continues_after_transport_disconnect_and_retry_replays_stream():
    async def run():
        calls = 0
        transport_disconnected = asyncio.Event()
        first_chunk_sent = asyncio.Event()
        finish = asyncio.Event()

        async def app(scope, receive, send):
            nonlocal calls
            _ = receive
            calls += 1
            detached_event = scope["state"][DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY]
            assert detached_event is not transport_disconnected
            assert not detached_event.is_set()
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
            await finish.wait()
            assert not detached_event.is_set()
            await send(
                {
                    "type": "http.response.body",
                    "body": b"data: done\n\n",
                    "more_body": False,
                }
            )

        coordinator = _coordinator()
        middleware = IdempotencyMiddleware(app, coordinator=coordinator)
        first_sent: list[dict] = []

        async def disconnecting_send(message):
            if transport_disconnected.is_set():
                raise OSError("client disconnected")
            first_sent.append(dict(message))

        first_task = asyncio.create_task(
            middleware(
                _scope(disconnect_event=transport_disconnected),
                _receive_body(b'{"model":"gpt-test"}'),
                disconnecting_send,
            )
        )
        await first_chunk_sent.wait()
        transport_disconnected.set()
        finish.set()
        await first_task

        replay = await _invoke(middleware)
        assert calls == 1
        assert _response(replay)[1] == b"data: first\n\ndata: done\n\n"
        assert _response(replay)[2][b"x-uni-api-idempotency-status"] == b"replayed"
        assert coordinator.snapshot()["downstream_disconnects_detached"] == 1

    asyncio.run(run())


def test_missing_key_preserves_the_original_hot_path():
    async def run():
        calls = 0

        async def app(scope, receive, send):
            nonlocal calls
            calls += 1
            assert any(
                name.lower() == b"content-type" for name, _value in scope["headers"]
            )
            await send(
                {"type": "http.response.start", "status": 204, "headers": []}
            )
            await send(
                {"type": "http.response.body", "body": b"", "more_body": False}
            )

        middleware = IdempotencyMiddleware(app, coordinator=_coordinator())
        await _invoke(middleware, scope=_scope(key=None))
        await _invoke(middleware, scope=_scope(key=None))
        assert calls == 2

    asyncio.run(run())


def test_idempotency_detachment_composes_with_admission_and_body_middleware():
    async def run():
        calls = 0
        first_chunk_sent = asyncio.Event()
        finish = asyncio.Event()
        original_disconnect = asyncio.Event()

        async def app(scope, receive, send):
            nonlocal calls
            calls += 1
            request = await receive()
            assert request["body"] == b'{"model":"gpt-test"}'
            await send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"a",
                    "more_body": True,
                }
            )
            first_chunk_sent.set()
            await finish.wait()
            await send(
                {
                    "type": "http.response.body",
                    "body": b"b",
                    "more_body": False,
                }
            )

        coordinator = _coordinator()
        controller = RequestAdmissionController(
            capacity=2,
            waiter_limit=2,
            wait_timeout_seconds=1,
            max_body_bytes=1024 * 1024,
            body_budget_bytes=4 * 1024 * 1024,
            max_response_bytes=1024 * 1024,
        )
        stack = RequestAdmissionMiddleware(
            IdempotencyMiddleware(
                RequestBodyDecompressionMiddleware(
                    app,
                    max_identity_body_bytes=1024,
                ),
                coordinator=coordinator,
            ),
            controller=controller,
        )
        disconnected_messages: list[dict] = []

        async def disconnecting_send(message):
            if original_disconnect.is_set():
                raise OSError("client disconnected")
            disconnected_messages.append(dict(message))

        owner_task = asyncio.create_task(
            stack(
                _scope(disconnect_event=original_disconnect),
                _receive_body(b'{"model":"gpt-test"}'),
                disconnecting_send,
            )
        )
        first_chunk_task = asyncio.create_task(first_chunk_sent.wait())
        done, _pending = await asyncio.wait(
            {owner_task, first_chunk_task},
            timeout=1.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if owner_task in done:
            owner_task.result()
        assert first_chunk_task in done
        original_disconnect.set()
        finish.set()
        await owner_task
        assert controller.snapshot()["active"] == 0

        replay_sent: list[dict] = []

        async def replay_send(message):
            replay_sent.append(dict(message))

        await stack(
            _scope(disconnect_event=asyncio.Event()),
            _receive_body(b'{"model":"gpt-test"}'),
            replay_send,
        )
        assert calls == 1
        assert _response(replay_sent)[1] == b"ab"
        assert controller.snapshot()["active"] == 0

    asyncio.run(run())
